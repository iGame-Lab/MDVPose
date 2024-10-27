import os
import h5py
import pickle
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--processing', default='data/motion3d/MB3D_f243s81/SKIPOSE', type=str, metavar='PATH', help="Path to save data.")
    parser.add_argument('-r', '--raw', default='', type=str, metavar='PATH', help='Path to raw data')
    opts = parser.parse_args()
    return opts

opts = parse_args()

os.makedirs(opts.processing, exist_ok=True)
os.makedirs(os.path.join(opts.processing, 'train'), exist_ok=True)
os.makedirs(os.path.join(opts.processing, 'test'), exist_ok=True)

h5_label_file = [[] for _ in range(2)]

h5_label_file[0] = h5py.File(os.path.join(opts.raw, 'train/labels.h5'), 'r')
h5_label_file[1] = h5py.File(os.path.join(opts.raw, 'test/labels.h5'), 'r')

data = [[] for _ in range(6)]
file_number = 0

def saving_and_padding(file, save_dir, file_number, data):
    #padding
    count = [[] for _ in range(6)]
    for cam_number in range(6):
        count[cam_number] = len(data[cam_number])
        count[cam_number] = 243 - count[cam_number]
        for _ in range(count[cam_number]):
            temp_data = data[cam_number][0].copy()
            temp_data["cam"] = 1024
            data[cam_number].append(temp_data)

    #saving
    if file == 0:   #train
        for i in range(6):
            with open(os.path.join(save_dir, "train/%08d_%01d.pkl" % (file_number, i)), 'wb') as f:
                pickle.dump(data[i], f)
    if file == 1:   #test
        for i in range(6):
            with open(os.path.join(save_dir, "test/%08d_%01d.pkl" % (file_number, i)), 'wb') as f:
                pickle.dump(data[i], f)

for file in range(2):
    temp_seq = int(h5_label_file[file]['seq'][0])
    for index in tqdm(range(len(h5_label_file[file]['3D']))):
        frame = int(h5_label_file[file]['frame'][index])
        seq = int(h5_label_file[file]['seq'][index])
        cam = int(h5_label_file[file]['cam'][index])
        pose_3D = h5_label_file[file]['3D'][index].reshape([-1,3])
        pose_2D_A = h5_label_file[file]['2D'][index].reshape([-1,2])
        pose_2D_B = pose_2D_A - 0.5
        confidence_2D = np.full((17, 1), 0.99999999, dtype=float)
        pose_2D_C = np.concatenate((pose_2D_B, confidence_2D), axis=1) 
        new_data = {
            "frame": frame,
            "seq": seq, 
            "cam": cam, 
            "pose_2D": pose_2D_C, 
            "pose_3D": pose_3D
        }

        if (temp_seq != seq):
            saving_and_padding(file, opts.processing, file_number, data)
            temp_seq = seq
            data.clear()
            data = [[] for _ in range(6)]
            file_number += 1
        
        data[cam].append(new_data)

    saving_and_padding(file, opts.processing, file_number, data)
    data.clear()
    data = [[] for _ in range(6)]
    file_number = 0 
    print("picture number =", index + 1)

