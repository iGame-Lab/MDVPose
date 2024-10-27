import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class skiposeDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(skiposeDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_data = {}
        # print("index = ", index, ", motion_file =", file_path)
        for item in motion_file:
            for key, value in item.items():
                if key not in motion_data:
                    motion_data[key] = []
                motion_data[key].append(value)
        
        if self.data_split=="train":
            motion_frame = np.squeeze(motion_data['frame'])
            motion_seq = np.squeeze(motion_data['seq'])
            motion_cam = np.squeeze(motion_data['cam'])
            motion_2d = np.squeeze(motion_data['pose_2D'])
            motion_3d = np.squeeze(motion_data['pose_3D'])
            # motion_c2w = np.squeeze(motion_data['R_cam_2_world'])
            if self.flip and random.random() > 0.5:
                motion_2d = flip_data(motion_2d)
                motion_3d = flip_data(motion_3d)
        elif self.data_split=="test":
            motion_frame = np.squeeze(motion_data['frame'])
            motion_seq = np.squeeze(motion_data['seq'])
            motion_cam = np.squeeze(motion_data['cam'])
            motion_2d = np.squeeze(motion_data['pose_2D'])
            motion_3d = np.squeeze(motion_data['pose_3D'])
            # motion_c2w = np.squeeze(motion_data['R_cam_2_world'])
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_frame), torch.FloatTensor(motion_seq), torch.FloatTensor(motion_cam), torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)

class egobodyDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(egobodyDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):

        def remove_non_numeric_chars(string):
            result = ""
            for char in string:
                if char.isdigit():
                    result += char
            return result
        
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_data = {}
        # print("index = ", index, ", motion_file =", file_path)
        for item in motion_file:
            for key, value in item.items():
                if key not in motion_data:
                    motion_data[key] = []
                motion_data[key].append(value)
        
        if self.data_split=="train":
            motion_frame = np.squeeze(motion_data['frame'])
            motion_seq = np.squeeze(motion_data['seq'])
            motion_cam = np.squeeze(motion_data['cam'])
            motion_2d = np.squeeze(motion_data['pose_2D'])
            motion_3d = np.squeeze(motion_data['pose_3D'])
            # if self.flip and random.random() > 0.5:
            #     motion_2d = flip_data(motion_2d)
            #     motion_3d = flip_data(motion_3d)
        elif self.data_split=="test":
            motion_frame = np.squeeze(motion_data['frame'])
            motion_seq = np.squeeze(motion_data['seq'])
            motion_cam = np.squeeze(motion_data['cam'])
            motion_2d = np.squeeze(motion_data['pose_2D'])
            motion_3d = np.squeeze(motion_data['pose_3D'])
        else:
            raise ValueError('Data split unknown.')
        motion_seq = [remove_non_numeric_chars(seq) for seq in motion_seq]
        return torch.FloatTensor(motion_frame), torch.DoubleTensor(list(map(np.double, motion_seq))), torch.FloatTensor(motion_cam), torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)

class threeDHPDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(threeDHPDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_data = {}
        # print("index = ", index, ", motion_file =", file_path)
        for item in motion_file:
            for key, value in item.items():
                if key not in motion_data:
                    motion_data[key] = []
                motion_data[key].append(value)
        
        if self.data_split=="train":
            motion_frame = np.squeeze(motion_data['index'])
            cam = np.squeeze(motion_data['cam'])      # only used in check
            scene = np.squeeze(motion_data['scene'])  # only used in check
            motion_3d = np.squeeze(motion_data['annot3'])
            motion_2d = np.squeeze(motion_data['annot2'])
            # if self.flip and random.random() > 0.5:
            #     motion_2d = flip_data(motion_2d)
            #     motion_3d = flip_data(motion_3d)
        elif self.data_split=="test":
            scale_3d = np.squeeze(motion_data['scale_3d'])
            motion_3d = np.squeeze(motion_data['annot3'])
            motion_2d = np.squeeze(motion_data['annot2'])
            return torch.FloatTensor(scale_3d), torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_frame), torch.FloatTensor(cam), torch.FloatTensor(scene), torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)

class H36MDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(H36MDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]
        frame = motion_file["index"]
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                # if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                #     motion_2d = flip_data(motion_2d)
                #     motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')
        cam = torch.FloatTensor(243).fill_(1)
        seq = torch.FloatTensor(243).fill_(1)
        return torch.FloatTensor(frame), seq, cam, torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)