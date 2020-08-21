"""Test a model and calcuate accuracy"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import utils

from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from args import get_test_args
from models import investorConferenceAnalyzer
from utils import Treebank, collate_fn

def main(args):
    # Set up logging
    args.save_dir = utils.get_save_dir(args.save_dir, args.name, training=False)
    log = utils.get_logger(args.save_dir, args.name)
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    device, args.gpu_ids = utils.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))

    # Get model
    log.info('Building model...')
    model = investorConferenceAnalyzer(args.pce_model, args.num_labels)
    model = nn.DataParallel(model, args.gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = utils.load_model(model, args.load_path, args.gpu_ids, return_step=False)
    model = model.to(device)
    model.eval()

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = Treebank(record_file)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    
    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = utils.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json.load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
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
        
        utils.plot_confusion_matrix(gold_dict, pred_dict, args.save_dir)
        results = utils.eval_dicts(gold_dict, pred_dict)
        results_list = [('NLL', nll_meter.avg),
                        ('Acc', results['Acc'])]
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        utils.visualize(tbx,
                        pred_dict=pred_dict,
                        eval_path=eval_file,
                        step=0,
                        split=args.split,
                        num_visuals=args.num_visuals)

if __name__ == '__main__':
    main(get_test_args())