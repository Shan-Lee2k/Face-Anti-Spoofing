import torch
import torch.utils.data
from .imagelist_dataset import ImageListDataset
from .df2dict_dataset import Df2DictDataset
from .casia_video_dataset import VideoDataset
from .casia_frame_dataset import FrameDataset
import time

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


class DatasetManager(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(dataset_config):
        if dataset_config.dataset_name == 'ImageListDataset':
            return ImageListDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'Df2DictDataset':
            return Df2DictDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'VideoDataset':
            return VideoDataset(dataset_config, dataset_config.transforms)
        elif dataset_config.dataset_name == 'FrameDataset':
            return FrameDataset(dataset_config, dataset_config.transforms)
        else:
            raise Exception('Unknown dataset type')

    @staticmethod
    def _get_sampler(sampler_config, dataset):
        if sampler_config.name == 'ClassProbability':
            dataset._create_index2class(sampler_config.class_column)
            needed_probs = sampler_config.class_probability
            class_probs = [0]*len(needed_probs)
            for index in range(len(dataset)):
                class_probs[dataset._get_class(index)] += 1
            class_probs = [x/sum(class_probs) for x in class_probs]
            weights = [x/y for x, y in zip(needed_probs, class_probs)]
            weights = torch.Tensor(weights)

            if sampler_config.num_elem_per_epoch is None:
                num_elements = len(dataset)
            elif type(sampler_config.num_elem_per_epoch) == int:
                num_elements = sampler_config.num_elem_per_epoch
            elif type(sampler_config.num_elem_per_epoch) == float:
                num_elements = int(sampler_config.num_elem_per_epoch * len(dataset))

            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_elements)
        elif sampler_config.name == 'NumElements':
            if type(sampler_config.num_elem_per_epoch) == int:
                num_elements = sampler_config.num_elem_per_epoch
            elif type(sampler_config.num_elem_per_epoch) == float:
                num_elements = int(sampler_config.num_elem_per_epoch * len(dataset))

            if num_elements > len(dataset):
                replacement = True
            else:
                replacement = False
                num_elements = None
            sampler = torch.utils.data.sampler.RandomSampler(dataset,
                                                             replacement,
                                                             num_elements)
        return sampler

    @staticmethod
    def get_dataloader(dataset_config, train_process_config, shuffle=True):
        t1=time.time()
        dataset = DatasetManager._get_dataset(dataset_config)
        t2=time.time()
        ex_time = t2 - t1
        print(f"Initialize and Transform dataset in {ex_time:.2f} s")
        if hasattr(dataset_config, 'sampler_config'):
            t1=time.time()
            sampler = DatasetManager._get_sampler(dataset_config.sampler_config, dataset)
            shuffle = False
            t2=time.time()
            ex_time = t2 - t1
            print(f"Sampling dataset in {ex_time:.2f} s")
        else:
            sampler = None
        t1=time.time()
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=train_process_config.batchsize,
                                                  shuffle=shuffle,
                                                  num_workers=train_process_config.nthreads,
                                                  sampler=sampler,
                                                  collate_fn= custom_collate_fn) # List samples from __getitem__
        t2=time.time()
        ex_time = t2 - t1
        print(f"Initialize dataloader in {ex_time:.2f} s")
        return data_loader

    @staticmethod
    def get_dataloader_by_args(dataset, batch_size, num_workers=8, shuffle=False, sampler=None):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  sampler=sampler)
        return data_loader
