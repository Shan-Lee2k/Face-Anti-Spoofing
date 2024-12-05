import argparse
import os
import torch
import torchvision as tv
from at_learner_core.utils import transforms
from at_learner_core.configs import dict_to_namespace
from at_learner_core.utils import transforms as transforms
import PIL
from at_learner_core.utils import joint_transforms as j_transforms
from at_learner_core.utils import sequence_transforms as s_transforms


pretrained_weights_dict = {
    "ImageNet_V2_Large": "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_large-5c1a4163.pth",
    "ImageNet_V1_Small": "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/at_learner_core/at_learner_core/models/architectures/mobilenet_v3_small-047dcff4.pth",
}
NUM_K = 2
L = 16
image_size = 112
modality_list = ["stat_r1000", "stat_r1"]
of_modality_list = ["optical_flow"]  # , 'optical_flow_start']
static_modality = ["random_static_image"]

test_seq_transform = tv.transforms.Compose(
    [
        s_transforms.LinspaceTransform(L, key_list=["data"]),
    ]
)

train_seq_transform = tv.transforms.Compose(
    [
        tv.transforms.RandomApply(
            [s_transforms.DuplicateElements(1, False, ["data"], "target", 1, True)],
            p=0.3,
        ),
        tv.transforms.RandomApply(
            [s_transforms.DuplicateElements(1, False, ["data"], "target", 0, False)],
            p=0.3,
        ),
        s_transforms.LinspaceTransform(L, key_list=["data"], max_start_index=0),
    ]
)

preprocess_transform = transforms.Transform4EachElement(
    [
        transforms.RemoveBlackBorders(),
        transforms.SquarePad(),
        tv.transforms.Resize(image_size),
    ]
)

postprocess_transform = tv.transforms.Compose(
    [
        transforms.CreateNewItem(transforms.RankPooling(C=1000), "data", "stat_r1000"),
        transforms.CreateNewItem(transforms.RankPooling(C=1), "data", "stat_r1"),
        transforms.DeleteKeys(["data"]),
        # transforms.DeleteKeys(['key_frame']),
        transforms.Transform4EachKey(
            [
                transforms.Transform4EachElement(
                    [
                        tv.transforms.ToTensor(),
                        tv.transforms.Resize(image_size),
                    ]
                ),
                transforms.StackTensors(squeeze=True),
                tv.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
            ],
            key_list=of_modality_list,
        ),
        transforms.Transform4EachKey(
            [
                tv.transforms.Resize(image_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5], std=[0.5]),
            ],
            key_list=modality_list,
        ),
        # TRANSFORM FOR MULTI STATIC IMAGE KMEANS
        # transforms.Transform4EachKey([
        #    transforms.ProcessImageList(tv.transforms.Compose([
        #        tv.transforms.Resize(256),
        #        tv.transforms.CenterCrop(224),
        #        tv.transforms.ToTensor(),
        #        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #    ]))
        # ], key_list=static_modality),
        # transforms.Transform4EachKey([
        #     tv.transforms.Resize(image_size),
        #     tv.transforms.ToTensor(),
        #     tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        #     #,
        #     ,key_list=static_modality),
        # TRANSFORM FOR SINGLE STATIC IMAGE
        transforms.Transform4EachKey(
            [
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ],
            key_list=static_modality,
        ),
    ]
)

train_image_transform = tv.transforms.Compose(
    [
        transforms.Transform4EachKey(
            [
                preprocess_transform,
                # tv.transforms.RandomApply(
                #     [j_transforms.RandomRotation((-5,5))], p= 1
                # ),
                
                tv.transforms.RandomApply(
                    [
                        transforms.RandomZoomWithResize(
                            (image_size - 6, image_size - 2),
                            target_size=(image_size, image_size),
                        )
                    ],
                    p=0.8,
                ),
                
                tv.transforms.RandomApply(
                    [j_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5
                ),
            ],
            key_list=["data"],
        ),
        transforms.Transform4EachKey(
            [
                # tv.transforms.RandomApply([j_transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.5),
                tv.transforms.RandomApply([
                    transforms.Transform4EachElement([
                        tv.transforms.RandomApply([
                            tv.transforms.RandomRotation(5)
                        ], p=0.5)
                    ])], p=0.5),
                tv.transforms.RandomApply(
                    [
                        transforms.Transform4EachElement(
                            [
                                tv.transforms.RandomApply(
                                    [
                                        tv.transforms.RandomCrop(
                                            image_size, padding=5, pad_if_needed=True
                                        )
                                    ],
                                    p=0.5,
                                )
                            ]
                        )
                    ],
                    p=0.5,
                ),
                tv.transforms.RandomApply(
                    [
                        transforms.Transform4EachElement(
                            [
                                tv.transforms.RandomApply(
                                    [tv.transforms.ColorJitter(0.05, 0.05, 0.05, 0.00)],
                                    p=0.5,
                                )
                            ]
                        )
                    ],
                    p=0.5,
                ),
            ],
            key_list=["data"],
        ),
        # Create static modality
        transforms.CreateNewItem(
            transforms.StaticImageTransform(L, "one"), "data", "random_static_image"
        ),
        # Create keyframe
        #transforms.CreateNewItem(transforms.KMeanKeyFrame(k=NUM_K), "data", "key_frame"),
        # Create OFTICAL FLOW
        transforms.CreateNewItem(
            transforms.LiuOpticalFlowTransform((0, 4), (L - 4, L)),
            "data",
            "optical_flow",
        ),
        # transforms.CreateNewItem(transforms.LiuOpticalFlowTransform(0,NUM_K -1), 'key_frame', 'optical_flow'),
        # transforms.CreateNewItem(transforms.LiuOpticalFlowTransform((0, 1), (2, 4)), 'data', 'optical_flow_start'),
        postprocess_transform,
    ]
)

test_image_transform = tv.transforms.Compose(
    [
        transforms.Transform4EachKey(
            [
                preprocess_transform,
            ],
            key_list=["data"],
        ),
        # Create keyframe
        #transforms.CreateNewItem(transforms.KMeanKeyFrame(k=2), "data", "key_frame"),
        # Create static modality
        transforms.CreateNewItem(
            transforms.StaticImageTransform(L, "one"), "data", "random_static_image"
        ),
        transforms.CreateNewItem(
            transforms.LiuOpticalFlowTransform(0, (L - 2, L)), "data", "optical_flow"
        ),
        #transforms.CreateNewItem(transforms.LiuOpticalFlowTransform(0, 1), 'key_frame', 'optical_flow'),
        postprocess_transform,
    ]
)
def get_config(protocol_name, batch_size=32, learning_rate=0.0001, THR = 0.5, nepochs=5, pretrained = None):
    config = {
        'head_config': {
            'task_name': 'rgb_track',
            'exp_name': f'exp1_{protocol_name}',
            'text_comment': '',
        },
        'checkpoint_config': {
            'out_path': None,
            'save_frequency': 1,
        },
        'datalist_config': {
            'trainlist_config': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '../data/train_list.txt',
                'protocol_name': protocol_name,
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'multi_static': None,
                'sampler_config': {
                    'name': 'NumElements',
                    'class_column': 'label',
                    'num_elem_per_epoch': 20.0,
                },
                'sequence_transforms': train_seq_transform,
                'transforms': train_image_transform,
            },
            'testlist_configs': {
                'dataset_name': 'VideoDataset',
                'datalist_path': '../data/dev_list.txt',
                'protocol_name': protocol_name,
                'data_columns': [('rgb_path', 'data')],
                'target_columns': ('label', 'target'),
                'group_column': 'video_id',
                'multi_static': None,
                'sequence_transforms': test_seq_transform,
                'transforms': test_image_transform,
            }
        },
        'train_process_config': {
            'nthreads': os.cpu_count(), #os.cpu_count(),
            'ngpu': 1,
            'batchsize': batch_size,
            'nepochs': nepochs,
            'resume': None,
            'optimizer_config': {
                'name': 'Adam',
                'lr_config': {
                    'lr_type': 'StepLR',
                    'lr': learning_rate,
                    'lr_decay_period':5,
                    'lr_decay_lvl': 0.5,
                },
                'weight_decay': 1e-05,
            },
        },
        'test_process_config': {
            'run_frequency': 1,
            'metric': {
                'name': 'acer',
                'target_column': 'target',
            }
        },
        'wrapper_config': {
            'wrapper_name': 'MultiModalWrapper',
            'input_modalities': modality_list + of_modality_list + static_modality,  # + static_modality, 
            'backbone': 'simplenet112',
            'backbone_static':'MobilenetV3', #'simplenet112',
            'nclasses': 1,
            'loss': 'BCE',
            'pretrained': pretrained,
        },
        'logger_config': {
            'logger_type': 'log_combiner',
            'loggers': [
                {'logger_type': 'terminal',
                 'log_batch_interval': 5,
                 'THR': THR,
                 'show_metrics': {
                     'name': 'acer',
                     'fpr': 0.01,
                 }},
            ]
        },
        'manual_seed': 42,
        'resume': None,
    }
    ns_conf = argparse.Namespace()
    dict_to_namespace(ns_conf, config)
    return ns_conf
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--savepath',
                        type=str,
                        default='experiments/',
                        help='Path to save options')
    parser.add_argument('--batchsize',
                   type=int,
                   default=16,
                   help='Batch size for training')
    parser.add_argument('--lr',
                   type= float,
                   default= 0.0001,
                   help='Learning rate for training')
    parser.add_argument('--thr',
                   type= float,
                   default= 0.5,
                   help='Threshold for testing')
    parser.add_argument('--nepochs',
                   type= int,
                   default= 5,
                   help='Epoch')
    parser.add_argument('--pretrained',
                   type= str or None,
                   default= None,
                   help='Pre-trained backbone for static modal')
    args = parser.parse_args()

    for idx in range(1, 4):
        configs = get_config(f'protocol_4_{idx}', args.batchsize, args.lr, args.thr, args.nepochs, args.pretrained)
        out_path = os.path.join(args.savepath,
                                configs.head_config.task_name,
                                configs.head_config.exp_name)
        os.makedirs(out_path, exist_ok=True)
        if configs.checkpoint_config.out_path is None:
            configs.checkpoint_config.out_path = out_path
        filename = os.path.join(out_path,
                                configs.head_config.task_name + '_' + configs.head_config.exp_name + '.config')

        torch.save(configs, filename)
        print('Options file was saved to ' + filename)
