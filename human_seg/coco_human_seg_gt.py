import cv2
import numpy as np

#part_ids = [ 0,  2,  4,  5,  6,  8, 13, 14, 16, 17, 18, 19, 20, 21, 24]
# the order is: (left right flipped)
# background, head, torso, left upper arm ,right upper arm, left forearm, right forearm,
#  left hand, right hand, left thigh, right thigh, left shank, right shank, left foot, right foot
part_ids = [0, 13, 2, 5, 8, 19, 20, 4, 24, 18, 6, 21, 16, 14, 17]

png_idx = [0, 14, 1, 11, 10, 13, 12, 2, 3, 6, 7, 8, 9, 5, 4]


def human_seg_spread_channel(human_seg_map):
    x = human_seg_map // 127
    x = x * np.array([9, 3, 1])
    x = np.add.reduce(x, 2)
    res = []
    for i in part_ids:
        res.append((x == i))
    res = np.stack(res, axis=-1)
    return res.astype(np.float32)

def human_seg_combine_channel(human_seg_split_map):
    segmap = np.add.reduce(human_seg_split_map * np.array(png_idx), 2)
    return np.stack([segmap], axis=-1).astype(np.uint8)

def human_seg_combine_argmax(human_seg_argmax_map):
    onehot = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in range(15)], axis=-1)
    return human_seg_combine_channel(onehot)