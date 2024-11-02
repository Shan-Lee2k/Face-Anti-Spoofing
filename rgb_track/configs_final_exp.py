import argparse
import os
import torch
import torchvision as tv
from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import transforms as transforms

from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms
from PIL import Image

pretrained_weights_dict = {
    'ImageNet_V2_Large': "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_large-5c1a4163.pth",
    'ImageNet_V1_Small':  "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_small-047dcff4.pth",
}

L = 16
image_size = 112
modality_list = ['stat_r1000', 'stat_r1']
of_modality_list = ['optical_flow'] #, 'optical_flow_start']
static_modality = ['random_static_image']

test_seq_transform = tv.transforms.Compose([
    s_transforms.LinspaceTransform(L, key_list=['data']),
])

train_seq_transform = tv.transforms.Compose([
    tv.transforms.RandomApply([
        s_transforms.DuplicateElements(1, False, ['data'], 'target', 1, True)
    ], p=0.5),
    s_transforms.LinspaceTransform(L, key_list=['data'], max_start_index=0),
])

preprocess_transform = transforms.Transform4EachElement([
    transforms.RemoveBlackBorders(),
    transforms.SquarePad(),
    tv.transforms.Resize(image_size),
])

postprocess_transform = tv.transforms.Compose([
    transforms.CreateNewItem(transforms.RankPooling(C=1000), 'data', 'stat_r1000'),
    transforms.CreateNewItem(transforms.RankPooling(C=1), 'data', 'stat_r1'),

    transforms.DeleteKeys(['data']),

    transforms.Transform4EachKey([
        transforms.Transform4EachElement([
            #tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
        ]),
        transforms.StackTensors(squeeze=True),
        tv.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
    ], key_list=of_modality_list),

    transforms.Transform4EachKey([
        tv.transforms.Resize(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5])],
        key_list=modality_list),
    
    transforms.Transform4EachKey([
        tv.transforms.Resize(112),
        tv.transforms.ToTensor(),
        ],
        #tv.transforms.Normalize(mean=[0.5], std=[0.5])],
        key_list=static_modality),
    
    
])

train_image_transform = tv.transforms.Compose([
    transforms.Transform4EachKey([
        preprocess_transform,
        
        tv.transforms.RandomApply([j_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5),
    ], key_list=['data']),
    
    
    transforms.Transform4EachKey([
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.RandomRotation(5)
                ], p=0.5)
            ])], p=0.5),
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.RandomCrop(image_size, padding=5, pad_if_needed=True)
                ], p=0.5)
            ])
        ], p=0.5),
        tv.transforms.RandomApply([
            transforms.Transform4EachElement([
                tv.transforms.RandomApply([
                    tv.transforms.ColorJitter(0.05, 0.05, 0.05, 0.00)
                ], p=0.5)
            ])
        ], p=0.5),
    ], key_list=['data']),
    
    # Create static modality
    transforms.CreateNewItem(transforms.StaticImageTransform(L), 'data', 'random_static_image'),
    
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 4), (L - 4, L)), 'data', 'optical_flow'),
    #transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 1), (2, 4)), 'data', 'optical_flow_start'),

    
    postprocess_transform

])

test_image_transform = tv.transforms.Compose([
    transforms.Transform4EachKey([
        preprocess_transform,
    ], key_list=['data']),
    
    # Create static modality
    transforms.CreateNewItem(transforms.StaticImageTransform(L), 'data', 'random_static_image'),
    transforms.CreateNewItem(transforms.LiuOpticalFlowTransform(0, L-1), 'data', 'optical_flow'),
    #transforms.CreateNewItem(transforms.LiuOpticalFlowTransform(0, 1), 'data', 'optical_flow_start'),

    
    postprocess_transform
])
