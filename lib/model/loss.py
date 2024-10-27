import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.utils.utils_icp import *
from lib.utils.procrust import *

def compute_valid_mean_norm(args, predicted, target, cam_number, dim_number):
    diff_pose = predicted - target
    # print(diff_pose.shape)
    mask = torch.tensor(cam_number < 100)
    selected_values = diff_pose[mask.unsqueeze(-1).unsqueeze(-1).expand_as(diff_pose)].view(-1, 17, 3)
    output_data = torch.mean(torch.norm(selected_values, dim=dim_number))
    return output_data

def clean_and_reduce_data(args, predicted, target, cam_number):
    vaild_pred = []
    vaild_gt = []
    index = np.zeros((args.batch_size,), dtype=int)

    for pred_outer, target_outer, cam_outer  in zip(predicted, target, cam_number):
        cam_number_temp = int(cam_outer[0])
        for pred_inner, target_inner, cam_inner in zip(pred_outer, target_outer, cam_outer):
            if cam_inner > 100:
                break
            index[cam_number_temp % args.batch_size] += 1
        pred_outer_temp = pred_outer[:int(index[cam_number_temp % args.batch_size])]
        target_outer_temp = target_outer[:int(index[cam_number_temp % args.batch_size])]
        vaild_pred.append(pred_outer_temp)
        vaild_gt.append(target_outer_temp)

    cated_tensor_pred = torch.cat(vaild_pred)
    cated_tensor_gt = torch.cat(vaild_gt)
    return cated_tensor_pred, cated_tensor_gt

# Numpy-based errors

def mpjpe(predicted, target, cam_item):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape

    # index  = (cam_item < 100).sum().item()    
    # pred = predicted[:index]
    # target = target[:index]

    mask = cam_item < 100
    pred = predicted[mask.unsqueeze(-1).unsqueeze(-1).expand_as(predicted)].view(-1, 17, 3)
    target = target[mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)].view(-1, 17, 3)

    return np.mean(np.linalg.norm(pred.cpu() - target.cpu(), axis=len(target.shape)-1), axis=1)

def mpjpe_2(predicted, target, cam_item):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape

    index  = (cam_item < 100).sum().item()    
    pred = predicted[:index]
    target = target[:index]

    return np.mean(np.linalg.norm(pred - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe(predicted, target, cam_item):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    target = target.cpu().numpy()
    predicted = predicted.cpu().numpy()
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE

    # index  = (cam_item < 100).sum().item()
    # pred = predicted_aligned[:index]
    # gt = target[:index]

    indices = np.where(cam_item < 100)
    pred = predicted_aligned[indices[0]]
    gt = target[indices[0]]
    return np.mean(np.linalg.norm(pred - gt, axis=len(gt.shape)-1), axis=1)

def mpjpe_author(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1), axis=1)

def p_mpjpe_author(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1), axis=1)

# PyTorch-based errors (for losses)

def loss_mpjpe(args, predicted, target, cam_number): 
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    dim_number = 2
    return compute_valid_mean_norm(args, predicted, target, cam_number, dim_number)

def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:,:,:,:2]
    target_2d = target[:,:,:,:2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))
    
def n_mpjpe(args, predicted, target, batch_cam):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(args, scale * predicted, target, batch_cam)

def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(args, x, cam_number):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    
    # previous code:
    # limb_lens_var = torch.var(limb_lens, dim=1)
    # limb_loss_var = torch.mean(limb_lens_var)

    output_data = []
    index = np.zeros((args.batch_size,), dtype=int)

    for limb_lens_outer, cam_outer  in zip(limb_lens, cam_number):
        cam_number_temp = int(cam_outer[0])
        for limb_lens_inner, cam_inner in zip(limb_lens_outer, cam_outer):
            if cam_inner > 100:
                break
            index[cam_number_temp % args.batch_size] += 1
        limb_lens_outer = limb_lens_outer[:int(index[cam_number_temp % args.batch_size])]
        output_data.append(torch.var(limb_lens_outer, dim=0))

    cated_tensor = torch.cat(output_data)
    weight_avg = torch.mean(cated_tensor)     
    return weight_avg

def loss_limb_gt(args, x, gt, batch_cam): 
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt) # (N, T, 16)
    pred_final, gt_final = clean_and_reduce_data(args, limb_lens_x, limb_lens_gt, batch_cam)
    return nn.L1Loss()(pred_final, gt_final)

def loss_velocity(args, predicted, target, cam_number):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    cam_number = torch.stack([cam[1:] for cam in torch.tensor(cam_number)])
    if predicted.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:,1:] - predicted[:,:-1]
    velocity_target = target[:,1:] - target[:,:-1]
    dim_number = -1
    return compute_valid_mean_norm(args, velocity_predicted, velocity_target, cam_number, dim_number)

def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [[0,1], [1,2], [2,3],
         [0,4], [4,5], [5,6],
         [0,7], [7,8], [8,9], [9,10],
         [8,11], [11,12], [12,13],
         [8,14], [14,15], [15,16]
        ]
    angle_id = [[ 0,  3],
                [ 0,  6],
                [ 3,  6],
                [ 0,  1],
                [ 1,  2],
                [ 3,  4],
                [ 4,  5],
                [ 6,  7],
                [ 7, 10],
                [ 7, 13],
                [ 8, 13],
                [10, 13],
                [ 7,  8],
                [ 8,  9],
                [10, 11],
                [11, 12],
                [13, 14],
                [14, 15] ]
    eps = 1e-7
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]
    angles = limbs[:,:,angle_id,:]
    angle_cos = F.cosine_similarity(angles[:,:,:,0,:], angles[:,:,:,1,:], dim=-1)
    return torch.acos(angle_cos.clamp(-1+eps, 1-eps)) 

def loss_angle(args, x, gt, batch_cam):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    pred_final, gt_final = clean_and_reduce_data(args, limb_angles_x, limb_angles_gt, batch_cam)
    return nn.L1Loss()(pred_final, gt_final)

def loss_angle_velocity(args, x, gt, batch_cam):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:,1:] - x_a[:,:-1]
    gt_av = gt_a[:,1:] - gt_a[:,:-1]
    pred_final, gt_final = clean_and_reduce_data(args, x_av, gt_av, batch_cam)
    return nn.L1Loss()(pred_final, gt_final)

def loss_multi_view(args, batch_frame, batch_seq, predicted_3d_pos, batch_gt, batch_cam):
    index = np.zeros((args.batch_size,), dtype=int)
    frame = sys.maxsize
    sub_2 = []

    for i in range(args.batch_size):
        frame = min(batch_frame[i][0], frame)
    
    for index_frame in range(243):
        same_frame_pose = []
        same_frame_pose_number = np.zeros((args.batch_size,), dtype=int)
        for index_cam in range(args.batch_size):
            if batch_cam[index_cam][index[index_cam]] > 100:
                continue
            if frame == batch_frame[index_cam][index[index_cam]]:
                same_frame_pose.append(predicted_3d_pos[index_cam][index[index_cam]])
                index[index_cam] += 1
                same_frame_pose_number[index_cam] = 1
        frame += 1
        
        first_one_index = np.argmax(same_frame_pose_number)
        for index_same_frame in range(args.batch_size):
            if same_frame_pose_number[index_same_frame] == 1 and first_one_index != index_same_frame:
                
                pred = predicted_3d_pos[index_same_frame][index_frame]
                pred_base = predicted_3d_pos[first_one_index][index_frame]

                R, s = procrustes(pred_base.detach().cpu().numpy(), pred.detach().cpu().numpy())
                temp = torch.matmul(pred, torch.tensor(R, dtype=torch.float32).to('cuda').T) * s
                sub_2.append(torch.sum(torch.abs(temp - pred_base)))

    return torch.mean(torch.stack(sub_2))
