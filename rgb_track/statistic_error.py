#Step 1: Read CSV -> Pandas -> (Sort 1->0)

#Step 2: Change predict based on thresold => (0,1)

#Step 3: Compare output vs label => TP,FP,TN,FN 
# Each TP,FP,TN,FN is dict with k:v === video_id
csv_path = "C:/Users/PC/Downloads/output_3.csv"
import pandas as pd
import pprint
data = pd.read_csv(csv_path, usecols= ["output_score", "video_id", "label"], skipinitialspace=True)
# Sort output score 1->0:
data["label"] = data["label"].str.extract(r'(\d+)$')
data.loc[0,"label"] = 1

data['label'] = data['label'].astype('int')
# Assuming your DataFrame is named 'df'

#data.sort_values(by=['output_score'])
def label_with_THR(dataframe,THR=0.15):
    dataframe['prediction'] = dataframe['output_score'].apply(lambda x: 1 if x >= THR else 0)
    return
def calculate_confusion_matrix(df, threshold=0.15):
    """
    Calculates TP, FP, TN, FN based on a given threshold.

    Args:
        df: DataFrame with columns `output_score`, `video_id`, and `label`.
        threshold: Threshold for classifying predictions.

    Returns:
        A tuple of four dictionaries, each containing video IDs for TP, FP, TN, and FN.
    """

    label_with_THR(df,threshold)
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    # tp = df[(df['output_score'] == 1) & (df['label'] == 1)]['video_id'].to_dict()
    # fp = df[(df['output_score'] == 1) & (df['label'] == 0)]['video_id'].to_dict()
    # tn = df[(df['output_score'] == 0) & (df['label'] == 0)]['video_id'].to_dict()
    # fn = df[(df['output_score'] == 0) & (df['label'] == 1)]['video_id'].to_dict()
    # Convert into dict 
    TP = data[(data['prediction'] == 1) & (data['label'] == 1)].set_index('video_id')['output_score'].to_dict()
    FP = data[(data['prediction'] == 1) & (data['label'] == 0)].set_index('video_id')['output_score'].to_dict()
    TN = data[(data['prediction'] == 0) & (data['label'] == 0)].set_index('video_id')['output_score'].to_dict()
    FN = data[(data['prediction'] == 0) & (data['label'] == 1)].set_index('video_id')['output_score'].to_dict()
    
    return TP,FP,TN,FN # DICTIONARY

def eval(df,threshold=0.15):
    TP,FP,TN,FN = calculate_confusion_matrix(df,threshold)
    APCER = len(FP)/(len(FP) + len(TN))
    BPCER = len(FN)/(len(TP) + len(FN))
    ACER = (APCER + BPCER) / 2 
    print(f"Test on THR = {threshold}")
    print(f"TP: {len(TP)}")
    print(f"FP: {len(FP)}")
    print(f"TN: {len(TN)}")
    print(f"FN: {len(FN)}")
    print(f"APCER: {APCER:.3f}")
    print(f"BPCER: {BPCER:.3f}")
    print(f"ACER: {ACER:.3f}")
    pass
def find_error_case(df, thr , case):
    TP,FP,TN,FN = calculate_confusion_matrix(df,thr)
    if case == "FP":
        pprint.pprint(FP)
    elif case == "FN":
        pprint.pprint(FN)
    else:
        raise TypeError("Please choosed correct keys FP or FN")
eval(data,threshold=0.1)
find_error_case(data,thr=0.15, case="FN")




    


