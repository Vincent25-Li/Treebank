"""Train a model on Treebank"""

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import utils

from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from args import get_train_args
from models import investorConferenceAnalyzer
from utils import Treebank, collate_fn

def main(args):
    # Set up logging and devices
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=True)
    log = utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = utils.get_available_devices()
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = investorConferenceAnalyzer(args.pce_model, args.num_labels)
    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.iofo(f'Loading checkpoint from {args.load_path}...')
        model, step = utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = utils.EMA(model, args.ema_decay)

    # Get saver
    saver = utils.CheckpointSaver(args.save_dir,
                                  max_checkpoints=args.max_checkpoints,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)
    
    # Get optimizer and scheduler
    optimizer_grouped_params = [
        {'params': model.module.classifier.albert.parameters()},
        {'params': model.module.classifier.classifier.parameters(), 'lr': args.lr_c}
    ]
    optimizer = optim.AdamW(optimizer_grouped_params, args.lr,
                            weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)

    # Get data loader
    log.info('Building dataset...')
    train_dataset = Treebank(args.train_record_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = Treebank(args.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_dataset)) as progress_bar:
            for input_idxs, token_type_idxs, attention_masks, ys, ids in train_loader:
                # Set up for forward
                input_idxs = input_idxs.to(device)
                token_type_idxs = token_type_idxs.to(device)
                attention_masks = attention_masks.to(device)
                batch_size = input_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                log_p = model(input_idxs, token_type_idxs, attention_masks)
                ys = ys.to(device)
                if args.smoothing:
                    loss = utils.nll_loss_label_smoothing(log_p, ys, args.eps)
                else:
                    loss = F.nll_loss(log_p, ys)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader,
                                                  device, args.dev_eval_file)

                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    utils.visualize(tbx,
                                    pred_dict=pred_dict,
                                    eval_path=args.dev_eval_file,
                                    step=step,
                                    split='dev',
                                    num_visuals=args.num_visuals)

def evaluate(model, data_loader, device, eval_file):
    nll_meter = utils.AverageMeter()

    model.eval()
    pred_dict = {}

    # Load eval info
    with open(eval_file, 'r') as fh:
        gold_dict = json.load(fh)

    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for input_idxs, token_type_idxs, attention_masks, ys, ids in data_loader:
            # Set up for forward
            input_idxs = input_idxs.to(device)
            token_type_idxs = token_type_idxs.to(device)
            attention_masks = attention_masks.to(device)
            batch_size = input_idxs.size(0)

            # Forward
            log_p = model(input_idxs, token_type_idxs, attention_masks)
            ys = ys.to(device)
            loss = F.nll_loss(log_p, ys)
            nll_meter.update(loss.item(), batch_size)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            # Get accuracy
            p = log_p.exp()
            labels = torch.argmax(p, dim=-1)
            preds = utils.predict_sentiments(ids.tolist(), labels.tolist())
            pred_dict.update(preds)
    
    model.train()

    results = utils.eval_dicts(gold_dict, pred_dict)
    results_list = [('NLL', nll_meter.avg),
                    ('Acc', results['Acc'])]
    results = OrderedDict(results_list)

    return results, pred_dict

if __name__ == '__main__':
    main(get_train_args())