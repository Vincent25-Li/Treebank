"""Utility classes and methods."""

import logging
import os
import queue
import shutil
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class Treebank(data.Dataset):
    """Treebank Dataset.
    Each item in the dataset is a tuple with the following entries (in order):
        - input_idx: Indices of the words in the text.
            Shape (2 + text_len,).
        - token_type_idx: Indices of the token type in the input.
            Shape (2 + text_max_len,).
        - attention_mask: Indices of the attention mask in the input.
            Shape (2 + text_max_len,).
        - y: Sentiment socre between [0, 4] of sentiment .
            -1 if no answer.
        - id: ID of the example.
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
    """
    def __init__(self, data_path):
        super(Treebank, self).__init__()

        dataset = np.load(data_path)
        self.input_idxs = torch.from_numpy(dataset['input_idxs']).long()
        self.token_type_idxs = torch.from_numpy(dataset['token_type_idxs']).long()
        self.attention_masks = torch.from_numpy(dataset['attention_masks']).long()
        self.ys = torch.from_numpy(dataset['ys']).long()
        self.ids = torch.from_numpy(dataset['ids']).long()

    def __getitem__(self, idx):
        example = (self.input_idxs[idx],
                   self.token_type_idxs[idx],
                   self.attention_masks[idx],
                   self.ys[idx],
                   self.ids[idx])

        return example

    def __len__(self):
        return len(self.ids)


def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `Treebank.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples of the form (input_idxs, token_type_idxs,
        attention_masks, ys, ids).
    Returns:
        examples (tuple): Tuple of tensors (input_idxs, token_type_idxs,
        attention_masks, ys, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_input_text(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded, max(lengths)
    
    def merge_1d(arrays, length, dtype=torch.int64):
        padded = torch.zeros(len(arrays), length, dtype=dtype)
        for i, seq in enumerate(arrays):
            padded[i] = seq[:length]
        return padded

    # Group by tensor type
    input_idxs, token_type_idxs, \
        attention_masks, ys, ids = zip(*examples)

    # Merge into batch tensors
    input_idxs, max_length = merge_input_text(input_idxs)
    token_type_idxs = merge_1d(token_type_idxs, max_length)
    attention_masks = merge_1d(attention_masks, max_length)
    ys = merge_0d(ys)
    ids = merge_0d(ids)

    return (input_idxs, token_type_idxs,
            attention_masks, ys, ids)

def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
        
    raise RuntimeError('Too many save directories crewated with the same name. \
                       Delete old save directories or use another name.')

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars."""
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    return device, gpu_ids

def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)
    
    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

def nll_loss_label_smoothing(log_p, targets, eps=0.1):
    """Calculate cross entropy loss with label smoothing

    Args:
        log_p (torch.Tensor): Log probability.
        targets (torch.Tensor): Target labels.
        eps (float): Label smoothing parameter.

    Returns:
        loss[torch.Tensor]: Mean entropy loss with label smoothing.
    """
    nll_loss = -log_p.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(-1)
    smooth_loss = -log_p.mean(dim=-1)
    loss = nll_loss * (1 - eps) + smooth_loss * eps
    return loss.mean()

class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avarage = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_avarage.clone()
    
    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True
        
        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))
        
    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')
        
        # Add checkpoint path to priority queue (lowesr priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val
        
        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass

class AverageMeter:
    """Keep track of average values over time."""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()
    
    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def predict_sentiments(ids, labels):
    """Convert probability to sentiment label.
    Args:
        eval_dict (dict): Dictionary with eval info for the dataset.
        labels (list): List of predicted sentiment labels.
        id (list): List of text IDs.
    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted label.
    """
    pred_dict = {}
    for id_, label in zip(ids, labels):
        pred_dict[str(id_)] = label
    return pred_dict

def eval_dicts(gold_dict, pred_dict):
    """Compute metrics
    Args:
        gold_dict (dict): Dictionary with eval info for the dataset.
        pred_dict (dict): Dictionary of predicted labels
    
    Returns:
        (dict): Dictionary of metrics
    """
    acc = total = 0
    for key, pred in pred_dict.items():
        total += 1
        ground_truth = gold_dict[key]['y']
        acc += compute_acc(pred, ground_truth)
    
    return {'Acc': 100. * acc / total}

def compute_acc(prediction, ground_truth):
    return int(prediction == ground_truth)

def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
    """Visualize text examples to TensorBoard

    Args:
        tbx (tensorboard.SummaryWriter): Summary writer.
        pred_dict (dict): Dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals < 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)
    
    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    with open(eval_path, 'r') as fh:
        eval_dict = json.load(fh)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_]
        example = eval_dict[id_]
        text = example['text']
        ground_truth = example['y']

        tbl_fmt = (f'- **Text:** {text}\n'
                   + f'- **Ground Truth:** {ground_truth}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)

def plot_confusion_matrix(gold_dict, pred_dict, base_path):
    true_labels = []
    pred_labels = []

    for idx, pred_label in pred_dict.items():
        true_labels.append(gold_dict[idx]['y'])
        pred_labels.append(pred_label)
    
    img_path = os.path.join(base_path, 'cm')
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    cm = confusion_matrix(true_labels, pred_labels)
    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\nprecision={precision:.3f}; recall={recall:.3f}; f1 score={f1:.3f}')
    plt.savefig(img_path, bbox_inches="tight")

def get_loss_fn(agrs):
    if args.binary:
        return F.binary_cross_entropy
    else:
        if args.smoothing:
            return nll_loss_label_smoothing
        else:
            return F.nll_loss
