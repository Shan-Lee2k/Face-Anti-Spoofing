
import torch
import cv2
from collections import OrderedDict
import torchvision as tv
from PIL import Image
from at_learner_core.datasets.dataset_manager import DatasetManager
from at_learner_core.datasets.casia_video_dataset import VideoDataset
import matplotlib.pyplot as plt
from at_learner_core.utils import transforms as transforms

from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms

PATH = "D:/SHAN/ĐACN/CASIA-SURF_CeFA/data/cefa/train/2_063_3_1_4/profile"

preprocess_transform = tv.transforms.Compose([
    transforms.RemoveBlackBorders(),
    transforms.SquarePad(),
    tv.transforms.Resize(112),
    #tv.transforms.RandomApply([tv.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.01),
])

post_transform = tv.transforms.Compose([
    tv.transforms.Resize(256),  # Resize to 256x256
    tv.transforms.CenterCrop(224),  # Crop to 224x224
    tv.transforms.ToTensor(),  # Convert to tensor with range [0, 1]
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

if __name__ == "__main__":
    #   GET CONFIG
    exp1 = "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/rgb_track/experiments/rgb_track/exp1_protocol_4_1/rgb_track_exp1_protocol_4_1.config"
    exp1_test = "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/rgb_track/experiment_tests/protocol_4_1/protocol_4_1.config"
    config = torch.load(exp1)
    config_test = torch.load(exp1_test)
    # TEST VIDEO DATASET 
    if hasattr(config.datalist_config.trainlist_config, 'multi_static'):
        print(config.datalist_config.trainlist_config.multi_static)
    train_set = VideoDataset(config.datalist_config.trainlist_config, config.datalist_config.trainlist_config.transforms)
    #print(train_set.df)
    #print(df)
    #print(train_set.__len__())
    
    
    
    # def rgb_loader(path):
    #     return Image.open(path).convert('RGB')
    # for _, column_name in train_set.data_columns: 
    #     print(f"Column_name: {column_name}")
    #     item_dict[column_name] = [rgb_loader(x) for x in item_dict[column_name]]
    
    # for _, column_name in train_set.target_columns:
    #     print(f"Column_name: {column_name}")
    #     item_dict[column_name] = torch.Tensor([item_dict[column_name]])
    # print(f" : {item_dict['data']}") 
    # if train_set.transforms is not None: 
    #     item_dict = train_set.transforms(item_dict)  
    # print(f" : {item_dict['key_frame']}") 

    
    

    
