import argparse
import os
import pandas as pd

import re



pattern = r"^(.*?)/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_path',
                        type=str,
                        default='./',
                        help='Path to your casia_surf cefa dataset')
    args = parser.parse_args()

    df = pd.read_csv('./train_list.txt')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df.to_csv('./train_list.txt', index=False)

    df = pd.read_csv('./dev_list.txt')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df.to_csv('./dev_list.txt', index=False)

    df = pd.read_csv('./dev_test_list_v1.txt')
    df['rgb_path'] = df.rgb_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['ir_path'] = df.ir_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df['depth_path'] = df.depth_path.apply(lambda x: os.path.join(args.data_path,re.match(pattern, x).group(1), x))
    df.to_csv('./dev_test_list.txt', index=False)

