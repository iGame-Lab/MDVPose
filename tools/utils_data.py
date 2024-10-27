import os
import torch
import torch.nn.functional as F
import numpy as np
import copy

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def crop_scale_3d(motion, scale_range=[1, 1]):
    '''
        Motion: [T, 17, 3]. (x, y, z)
        Normalize to [-1, 1]
        Z is relative to the first frame's root.
    '''
    result = copy.deepcopy(motion)
    result[:,:,2] = result[:,:,2] - result[0,0,2]
    xmin = np.min(motion[...,0])
    xmax = np.max(motion[...,0])
    ymin = np.min(motion[...,1])
    ymax = np.max(motion[...,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) / ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,2] = result[...,2] / scale
    result = (result - 0.5) * 2
    return result

def normalization_3D_pose(motion):
    result = copy.deepcopy(motion)
    xmin = np.percentile(motion[...,0], 1)
    xmax = np.percentile(motion[...,0], 99)
    ymin = np.percentile(motion[...,1], 1)
    ymax = np.percentile(motion[...,1], 99)
    zmin = np.percentile(motion[...,2], 1)
    zmax = np.percentile(motion[...,2], 99)
    scale = max(xmax-xmin, ymax-ymin, zmax-zmin)
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    zs = (zmin+zmax-scale) / 2
    result = (motion - [xs,ys,zs]) / scale
    result = (result - 0.5) * 2
    return result, scale

def normalization_2D_pose(motion):
    result = copy.deepcopy(motion)
    xmin = np.percentile(motion[...,0], 0)
    xmax = np.percentile(motion[...,0], 100)
    ymin = np.percentile(motion[...,1], 0)
    ymax = np.percentile(motion[...,1], 100)
    scale = max(xmax-xmin, ymax-ymin)
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result = (motion - [xs,ys]) / scale
    result = (result - 0.5) * 2
    return result, scale

def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1                                               # flip x of all joints
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

def split_clips(vid_list, n_frames, data_stride):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i<len(vid_list):
        i += 1
        if i-st == n_frames:
            result.append(range(st,i))
            saved.add(vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len(vid_list):
            break
        if vid_list[i]!=vid_list[i-1]: 
            if not (vid_list[i-1] in saved):
                resampled = resample(i-st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i-1])
            st = i
    return result


if __name__ == "__main__":
    motion = np.array(
     [[[ 437.373     , -305.874     , 3899.32      ],
       [ 422.528     ,  -50.5686    , 3913.74      ],
       [ 437.353     ,   22.2367    , 3786.54      ],
       [ 433.999     ,  109.698     , 3472.25      ],
       [ 468.79      ,  167.101     , 3229.2       ],
       [ 405.804     ,   10.7749    , 4047.67      ],
       [ 353.592     ,   18.9484    , 4369.62      ],
       [ 336.221     ,   37.5931    , 4620.47      ],
       [ 484.533     ,  427.1645    , 3819.7525    ],
       [ 486.928     ,  963.369     , 3838.3       ],
       [ 350.553     , 1369.96      , 3867.9       ],
       [ 452.445     ,  420.7835    , 4060.0075    ],
       [ 441.115     ,  958.501     , 4032.15      ],
       [ 321.704     , 1369.73      , 3994.43      ],
       [ 468.489     ,  423.974     , 3939.88      ],
       [ 434.71796675,  202.79777301, 3927.70605991],
       [ 438.83      , -134.715     , 3910.99      ]],
      [[ 434.754     , -314.377     , 3904.83      ],
       [ 432.956     ,  -58.9991    , 3907.43      ],
       [ 469.46      ,    6.15894   , 3783.55      ],
       [ 448.75      ,  113.042     , 3476.        ],
       [ 473.145     ,  173.36      , 3232.39      ],
       [ 392.395     ,   -7.54749   , 4041.75      ],
       [ 348.799     ,   21.8706    , 4363.73      ],
       [ 336.42      ,   57.683     , 4613.01      ],
       [ 504.5585    ,  413.72325   , 3820.3625    ],
       [ 480.539     ,  951.518     , 3841.13      ],
       [ 363.785     , 1363.88      , 3874.53      ],
       [ 419.1395    ,  418.69875   , 4047.2375    ],
       [ 428.001     ,  955.263     , 4033.14      ],
       [ 324.675     , 1369.73      , 3984.87      ],
       [ 461.849     ,  416.211     , 3933.8       ],
       [ 433.35734972,  194.2500637 , 3921.57322267],
       [ 448.286     , -143.373     , 3907.69      ]]])
    ans = normalization_2D_pose(motion[...,:2])