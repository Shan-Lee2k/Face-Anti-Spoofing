import argparse
import os
import torch
import torchvision as tv
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import sequence_transforms as s_transforms

def get_config(protocol_name, THR = 0.5):
    test_config = {
        'test_config_name': protocol_name,
        'out_path': None,
        'ngpu': 1,
        'dataset_configs': {
            'dataset_name': 'VideoDataset',
            'datalist_path': '../data/dev_test_list_v1.txt',
            'data_columns': [('rgb_path', 'data')],
            'target_columns': ('label', 'target'),
            'transform_source': 'model_config',
            'group_column': 'video_id',
            'protocol_name': protocol_name,
            'seq_transform_source': 'model_config',
            'multi_static': None,
            'test_process_config': {
                'metric': {
                    'name': 'acer',
                    'target_column': 'target'
                }
            },
            'nthreads': os.cpu_count(),
            'batch_size': 32,
        },

        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'test_filelogger',
                 'show_metrics': {
                     'name': 'acer',
                 },
                 },
                {'logger_type': 'test_terminal',
                 'THR': THR,
                 'show_metrics': {
                     'name': 'acer',
                 },
                 }
            ],
        },
    }

    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, test_config)
    return ns_conf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiment_tests/',
                        help='Path to save options')
    parser.add_argument('--thr',
                   type= float,
                   default= 0.5,
                   help='Threshold for testing')
    args = parser.parse_args()
    for idx in range(1, 4):
        configs = get_config(f'protocol_4_{idx}',args.thr)
        out_path = os.path.join(args.savepath,
                                configs.test_config_name)
        os.makedirs(out_path, exist_ok=True)
        if configs.out_path is None:
            configs.out_path = out_path
        filename = os.path.join(out_path,
                                configs.test_config_name + '.config')

        torch.save(configs, filename)
        print('Options file was saved to ' + filename)
