import os
import sys
import pickle
import numpy as np
import random
sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_multiview_h36m import DataReaderH36M_MV
from tqdm import tqdm


def save_clips(subset_name, root_path, train_data, train_labels, filename):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    index = np.arange(1, 244)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "index": index,
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, filename[i]), "wb") as myprofile:  
            pickle.dump(data_dict, myprofile)
            
datareader = DataReaderH36M_MV(n_frames=243, sample_stride=1, data_stride_train=80, data_stride_test=243, dt_file = 'h36m_sh_conf_cam_source_final.pkl', dt_root='data/motion3d')
train_data, test_data, train_labels, test_labels, filename_train, filename_test = datareader.get_sliced_data()
# train_data[:,:,0,:] = train_data[:,:,:,:] - train_data[:,:,0,:]

first_row_first_two = train_data[:, :, 0, :2]
train_data[:, :, :, :2] -= first_row_first_two[:, :, np.newaxis, :]

first_row_first_two = test_data[:, :, 0, :2]
test_data[:, :, :, :2] -= first_row_first_two[:, :, np.newaxis, :]

first_row = train_labels[:, :, 0]
train_labels -= first_row[:, :, np.newaxis]

first_row = test_labels[:, :, 0]
test_labels -= first_row[:, :, np.newaxis]

print(train_data.shape, test_data.shape)
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

# for i in test_data:
#     i[-40:, :, :] = 0
# for i in train_data:
#     i[-40:, :, :] = 0

root_path = "data/motion3d/MB3D_f243s81/H36M2"
if not os.path.exists(root_path):
    os.makedirs(root_path)

save_clips("train", root_path, train_data, train_labels, filename_train)
save_clips("test", root_path, test_data, test_labels, filename_test)

