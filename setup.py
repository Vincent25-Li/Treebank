"""Pre-process Treebank data"""

import re
import os
import json
import numpy as np

from tqdm import tqdm
from transformers import AlbertTokenizer
from args import get_setup_args

def save(filename, obj, message=None):
    if message is not None:
        print(f'Saving {message}...')
    
    with open(filename, "w") as fh:
            json.dump(obj, fh)

def process_file(filename, data_type, binary=False):
    """Process file and collect examples"""
    print(f'Pre-processing {data_type} examples...')
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, 'r') as fp:
        for line in tqdm(fp.readlines()):
            if binary:
                if y := int(line[1]) != 2:
                    total += 1 
                    pattern = r"\(\d ([^\(\)]*?)\)"
                    text_tokens = re.findall(pattern, line)
                    text = ' '.join(text_tokens)
                    y = int(y > 2)
                    example = {'text': text,
                            'y': y,
                            'id': total}
                    examples.append(example)
                    eval_examples[str(total)] = {'text': text, 'y': y}
                else:
                    continue
            else:
                total += 1    
                pattern = r"\(\d ([^\(\)]*?)\)"
                text_tokens = re.findall(pattern, line)
                text = ' '.join(text_tokens)
                y = int(line[1])
                example = {'text': text,
                        'y': y,
                        'id': total}
                examples.append(example)
                eval_examples[str(total)] = {'text': text, 'y': y}
            
    return examples, eval_examples

def build_features(args, examples, data_type, out_file, tokenizer):
    """Build features for models"""
    print(f'Convert {data_type} examples to input features...')
    meta = {}
    input_idxs = []
    token_type_idxs = []
    attention_masks = []
    ys = []
    ids = []
    for example in tqdm(examples):
        # Convert text into tokens
        text = example['text']

        encode_dict = tokenizer.encode_plus(text,
                                            padding='max_length',
                                            max_length=args.max_len,
                                            truncation=True)
        input_idx = encode_dict['input_ids']
        token_type_idx = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']

        input_idxs.append(input_idx)
        token_type_idxs.append(token_type_idx)
        attention_masks.append(attention_mask)
        
        ys.append(example['y'])
        ids.append(example['id'])
    
    # Record meta data
    meta['total'] = len(ids)

    np.savez(out_file,
             input_idxs=np.array(input_idxs),
             token_type_idxs=np.array(token_type_idxs),
             attention_masks=np.array(attention_masks),
             ys=np.array(ys),
             ids=np.array(ids))
    
    print(f'Build {len(ids)} {data_type} data')

    return meta

def pre_process(args):
    # Load tokenizer
    print('Loading tokenizer...')
    tokenizer = AlbertTokenizer.from_pretrained(args.pce_model)

    # Process train, dev, and test file
    train_examples, train_eval = process_file(args.train_data_path, 'train', binary=args.binary)
    dev_examples, dev_eval = process_file(args.dev_data_path, 'dev', binary=args.binary)
    test_examples, test_eval = process_file(args.test_data_path, 'test', binary=args.binary)

    def get_max_len(args, examples, tokenizer):
        max_len = max([len(tokenizer.encode(example['text'])) for example in examples])
        return max_len if max_len <= args.max_input_len else args.max_input_len

    args.max_len = get_max_len(args, train_examples, tokenizer) 

    train_meta = build_features(args, train_examples, 'train',
                                args.train_record_file, tokenizer)
    dev_meta = build_features(args, dev_examples, 'dev',
                              args.dev_record_file, tokenizer)
    test_meta = build_features(args, test_examples, 'test',
                               args.test_record_file, tokenizer)

    save(args.train_eval_file, train_eval, message='train eval')
    save(args.dev_eval_file, dev_eval, message='dev eval')
    save(args.test_eval_file, test_eval, message='test eval')
    save(args.train_meta_file, train_meta, message='train meta')
    save(args.dev_meta_file, dev_meta, message='dev meta')
    save(args.test_meta_file, test_meta, message='test meta')
    
if __name__ == '__main__':
    pre_process(get_setup_args())