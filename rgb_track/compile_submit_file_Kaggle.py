import pandas as pd

if __name__ == '__main__':
    submit_file_df = pd.DataFrame()
    for idx in range(1, 4):
        v_dev = pd.read_csv(f'../data/4_{idx}_dev_res.txt', names=['video_id'])
        v_test = pd.read_csv(f'../data/4_{idx}_test_res.txt', names=['video_id'])
        df = pd.read_csv('/kaggle/input/cefa-nopass/output_4_bs64.csv')
        df.index = df.video_id
        v = pd.concat([v_dev, v_test])
        v.index = range(v.index.size)
        v['score'] = df.loc[v.video_id.values].output_score.values
        submit_file_df = submit_file_df.append(v, ignore_index=True)

    submit_file_df.to_csv('./submit_file.txt', index=False, header=False, sep=' ')