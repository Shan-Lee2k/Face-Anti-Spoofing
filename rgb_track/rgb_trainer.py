from at_learner_core.trainer import Runner
from models.wrappers.rgb_simple_wrapper import RGBSimpleWrapper
from models.wrappers.rgb_transformer_wrapper import RGBVideoWrapper
from models.wrappers.multi_modal_wrapper import MultiModalWrapper
from models.wrappers.sdnet_wrapper import SDNetWrapper
from models.wrappers.dlas_wrapper import DLASWrapper
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from at_learner_core.datasets.dataset_manager import DatasetManager

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable sample sizes in a batch.
    
    Parameters:
    - batch: list of lists, where each sublist contains either 5 or 1 item_dict
    
    Returns:
    - batched_dict: dictionary where each key has a batch of stacked items, maintaining batch_size.
    """
    # Initialize lists for each key in item_dicts to store batched data
    batched_dict = {key: [] for key in batch[0][0].keys()}
    
    cumulative_samples = 0
    batch_size = 32  # Define your target batch size
    
    for item_list in batch:
        sample_count = len(item_list)
        
        # Stop adding samples if cumulative_samples reaches batch_size
        if cumulative_samples + sample_count > batch_size:
            break
        
        # Append items to the batched_dict keys
        for item_dict in item_list:
            for key, value in item_dict.items():
                batched_dict[key].append(value)
        
        cumulative_samples += sample_count

    # Stack the lists into tensors if needed, adjust per your requirement
    for key, values in batched_dict.items():
        batched_dict[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    
    return batched_dict


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class RGBRunner(Runner):
    def __init__(self, config, train=True):
        super().__init__(config, train=train)

    def _init_wrapper(self):
        if self.config.wrapper_config.wrapper_name == 'RGBSimpleWrapper':
            self.wrapper = RGBSimpleWrapper(self.config.wrapper_config)
        elif self.config.wrapper_config.wrapper_name == 'RGBVideoWrapper':
            self.wrapper = RGBVideoWrapper(self.config.wrapper_config)
        elif self.config.wrapper_config.wrapper_name == 'MultiModalWrapper':
            self.wrapper = MultiModalWrapper(self.config.wrapper_config)
        elif self.config.wrapper_config.wrapper_name == 'DLASWrapper':
            self.wrapper = DLASWrapper(self.config.wrapper_config)
        elif self.config.wrapper_config.wrapper_name == 'SDNetWrapper':
            self.wrapper = SDNetWrapper(self.config.wrapper_config)

        if hasattr(self.config.wrapper_config, 'freeze_weights') and self.config.wrapper_config.freeze_weights:
            import re
            if type(self.config.wrapper_config.freeze_weights) == str:
                regexes = [self.config.wrapper_config.freeze_weights]
            else:
                regexes = self.config.wrapper_config.freeze_weights

            for param_name, param in self.wrapper.named_parameters():
                for regex in regexes:
                    if re.search(regex, param_name):
                        param.requires_grad = False
                        break

        self.wrapper = self.wrapper.to(self.device)

    def _init_loaders(self):
        self.train_loader = DatasetManager.get_dataloader(self.config.datalist_config.trainlist_config,
                                                          self.config.train_process_config,
                                                         custom_data = None)

        self.val_loader = DatasetManager.get_dataloader(self.config.datalist_config.testlist_configs,
                                                        self.config.train_process_config,
                                                        shuffle=False,
                                                       custom_data = None)
if __name__  == '__main__':
    torch.set_printoptions(edgeitems=1000)
    writer = SummaryWriter('runs/FAS_experiment_1')
    exp1 = "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/rgb_track/experiments/rgb_track/exp1_protocol_4_1/rgb_track_exp1_protocol_4_1.config"
    exp1_test = "C:/Users/PC/Documents/GitHub/Face-Anti-Spoofing/rgb_track/experiment_tests/protocol_4_1/protocol_4_1.config"
    config = torch.load(exp1)
    config_test = torch.load(exp1_test) 
    #print(config)
    #dataset = DatasetManager._get_dataset(config.datalist_config.trainlist_config)
    train_loader = DatasetManager.get_dataloader(config.datalist_config.trainlist_config, config.train_process_config)
    
    dataset_config = config_test.dataset_configs
    if dataset_config.transform_source == 'model_config':
        transforms = config.datalist_config.testlist_configs.transforms
        setattr(dataset_config, 'transforms', transforms)    

    if dataset_config.seq_transform_source == 'model_config':
        seq_transforms = config.datalist_config.testlist_configs.sequence_transforms
        setattr(dataset_config, 'sequence_transforms', seq_transforms)

    test_dataset = DatasetManager._get_dataset(config_test.dataset_configs)
    test_loader = DatasetManager.get_dataloader_by_args(dataset=test_dataset,batch_size=16,num_workers=4)
    #test_set = DatasetManager._get_dataset(config.datalist_config.testlist_configs)
    # test_loader = DatasetManager.get_dataloader(config.datalist_config.testlist_configs,
    #                                                     config.train_process_config,
    #                                                     shuffle=True)
    
    
    # get some random training images
    dataiter = iter(test_loader)
    data = next(dataiter)
    target, optical_flow, random_static_image, stat_r1000, stat_r1, video_id = data['target'], data['optical_flow'], data['random_static_image'], data['stat_r1000'], data['stat_r1'], data['video_id']
    images =  [stat_r1000, stat_r1]
    ran_op = [random_static_image, optical_flow]
    
    print(optical_flow.size())
    print(random_static_image.size())
    print(stat_r1000.size())
    print(stat_r1.size())
    print(random_static_image[0])
    print('STATIC')
    # for i in range(8):
    #     image = random_static_image[i]
    #     plt.imshow(image.permute(1,2,0))
    #     plt.title(video_id[i])
    #     plt.show()
    # print('RANK POOLING 1000')
    # # RANK POOLING
    # for i in range(8):
    #     image = stat_r1000[i]
    #     plt.imshow(image.permute(1,2,0))
    #     plt.title(video_id[i])
    #     plt.show()
        
    # print('RANK POOLING 1')
    # # RANK POOLING
    # for i in range(8):
    #     image = stat_r1[i]
    #     plt.imshow(image.permute(1,2,0))
    #     plt.title(video_id[i])
    #     plt.show()
    # print('OPTICAL FLOW')
    # # OPTICAL FLOW    
    for i in range(16):
        static = random_static_image[i]
        rank_1 = stat_r1[i]
        rank_1000  =  stat_r1000[i]
        #optical_flow = optical_flow[i]
        # Denormalize:
        for img in [rank_1,rank_1000,optical_flow]:
            img = img / 2 + 0.5
        
        #x_flow = optical_flow[0]
        #y_flow = optical_flow[1]
        #print(f"X.shape: {x_flow.size()} ")
        #print(f"Y.shape: {y_flow.size()} ")
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(static.permute(1,2,0))
        plt.title(f'Static image\n {video_id[i]}')

        plt.subplot(1, 3, 2)
        plt.imshow(rank_1.permute(1,2,0))
        plt.title(f'Rank Pooling C=1 \n {video_id[i]}')
        
        plt.subplot(1, 3, 3)
        plt.imshow(rank_1000.permute(1,2,0))
        plt.title(f'Rank Pooling C=1000 \n {video_id[i]}')
        
        # plt.subplot(1, 5, 4)
        # plt.imshow(x_flow, cmap = 'gray')
        
        # plt.title(f'Optical flow X \n {video_id[i]}')
        
        # plt.subplot(1, 5, 5)
        # plt.imshow(y_flow, cmap = 'gray')

        # plt.title(f'Optical flow Y \n {video_id[i]}')
        

        plt.show()
