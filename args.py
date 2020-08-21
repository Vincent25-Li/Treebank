import argparse

def get_setup_args():
    """Get arguments needed in setup.py"""
    parser = argparse.ArgumentParser('Pre-process Treebank data')

    add_common_args(parser)

    parser.add_argument('--train_data_path',
                        type=str,
                        default='./data/trees/train.txt')
    parser.add_argument('--dev_data_path',
                        type=str,
                        default='./data/trees/dev.txt')
    parser.add_argument('--test_data_path',
                        type=str,
                        default='./data/trees/test.txt')
    parser.add_argument('--train_meta_file',
                        type=str,
                        default='./data/train_meta.json')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/test_meta.json')
    parser.add_argument('--max_input_len',
                        type=int,
                        default=512,
                        help='Maximum input length of textsval_file')
    
    args = parser.parse_args()

    if args.binary:
        args = change_binary_file_path(args)

    return args

def get_train_args():
    """Get arguments needed in train.py"""
    parser = argparse.ArgumentParser('Train a model on Treebank dataset')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--seed',
                        type=int,
                        default=995,
                        help='Random seed for reproducibility.')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.999,
                        help='Decay rate for exponential moving average of parameters.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='Acc',
                        choices=('NLL', 'Acc'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='Default learning rate.')
    parser.add_argument('--lr_c',
                        type=float,
                        default=1e-5,
                        help='Learning rate for classifier layer.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0.01,
                        help='L2 weight decay.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=6000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    
    args = parser.parse_args()
    
    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name == 'Acc':
        # Best checkpoint is the one that maximizes F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')
    
    if args.binary:
        args.num_labels = 2

    return args

def get_test_args():
    """Get arguments need in test.py"""
    parser = argparse.ArgumentParser('Test a trained model on Treebank')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--split',
                        type=str,
                        default='dev',
                        choices=('train', 'dev', 'test'))

    args = parser.parse_args()
    
    if args.binary:
        args.num_labels = 2

    if args.load_path is None:
        raise argparse.ArgumentError('Missing required argument --load_path')

    return args

def add_common_args(parser):
    """Add common arguments to scripts: setup.py, train.py, and test.py"""
    parser.add_argument('--pce_model',
                        type=str,
                        default='albert-base-v2',
                        help='Pre-trained contextual embedding used.')
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')
    parser.add_argument('--binary',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to analyze binary sentiment classification.')


def add_train_test_args(parser):
    """Add arguments to train.py and test.py"""
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        required=True,
                        help='Name to identify training or test run.')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size per GPU. Scales automatically \
                             when multiple GPUs are available.')
    parser.add_argument('--num_labels',
                        type=int,
                        default=5,
                        help='Number of sentiment labels.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--num_visuals',
                        type=int,
                        default=10,
                        help='Number of examples to visualize in TensorBoard.')
    parser.add_argument('--smoothing',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to apply label smoothing.')
    parser.add_argument('--eps',
                        type=float,
                        default=0.1,
                        help='Label smoothing parameter.')

def change_binary_file_path(args):
    """Change file names for binary sentiment classification data

    Args:
        args (object): Arguments needed.
    """
    changed_files = ('record_file', 'eval_file', 'meta_file')
    for key, value in vars(args).items():
        if any(name in key for name in changed_files):
            value = vars(args)[key]
            path = value.split(".")
            path[-2] = f'{path[-2]}_b'
            value = '.'.join(path)
            setattr(args, key, value)
    
    return args