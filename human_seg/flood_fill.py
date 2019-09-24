import numpy as np
import sys

sys.setrecursionlimit(100000)

def parse_keypoints(keypoints_str):
    keypoints = keypoints_str.rstrip().split('\n')
    keypoints = list(map(lambda x: list(map(lambda y: list(map(lambda z: int(float(z)), y.rstrip().split(','))), x.rstrip().split(' '))), keypoints))
    return keypoints

def check_depth(coord, coord_prev, depth_map, depth_diff_thres):
    if np.abs(float(depth_map[tuple(coord)]) - float(depth_map[tuple(coord_prev)])) < depth_diff_thres:
        return True
    return False

def check_keypoint_idx(coord, keypoint_idx, seg_map):
    # no need to land on correct parts
    if np.min(seg_map[tuple(coord)] == np.array([0, 0, 0])):
        return False
    return True


    if keypoint_idx in [0, 14, 15, 16, 17] and np.min(seg_map[tuple(coord)] == np.array([127, 127, 127])):
        return True
    if keypoint_idx in [1] and (np.min(seg_map[tuple(coord)] == np.array([127, 127, 127])) or np.min(seg_map[tuple(coord)] == np.array([0, 0, 255]))):
        return True
    if keypoint_idx in [2, 5] and (np.min(seg_map[tuple(coord)] == np.array([0, 255, 255])) or np.min(seg_map[tuple(coord)] == np.array([0, 0, 255])) or np.min(seg_map[tuple(coord)] == np.array([0, 127, 255]))):
        return True
    if keypoint_idx in [3, 6] and (np.min(seg_map[tuple(coord)] == np.array([0, 255, 255])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 255])) or np.min(seg_map[tuple(coord)] == np.array([0, 127, 255])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 127]))):
        return True
    if keypoint_idx in [4, 7] and (np.min(seg_map[tuple(coord)] == np.array([255, 255, 0])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 255])) or np.min(seg_map[tuple(coord)] == np.array([0, 127, 127])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 127]))):
        return True
    if keypoint_idx in [8, 11] and (np.min(seg_map[tuple(coord)] == np.array([0, 0, 255])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 0])) or np.min(seg_map[tuple(coord)] == np.array([0, 255, 0]))):
        return True
    if keypoint_idx in [9, 12] and (np.min(seg_map[tuple(coord)] == np.array([0, 255, 0])) or np.min(seg_map[tuple(coord)] == np.array([127, 255, 127])) or np.min(seg_map[tuple(coord)] == np.array([255, 0, 0])) or np.min(seg_map[tuple(coord)] == np.array([255, 127, 0]))):
        return True
    if keypoint_idx in [10, 13] and (np.min(seg_map[tuple(coord)] == np.array([127, 255, 127])) or np.min(seg_map[tuple(coord)] == np.array([127, 255, 255])) or np.min(seg_map[tuple(coord)] == np.array([255, 127, 0])) or np.min(seg_map[tuple(coord)] == np.array([127, 127, 255]))):
        return True
    return False

# coordinate should be transposed
def flood_fill(target_map, check_map, coord, coord_prev, keypoint_idx, target_map_instance_id, seg_map, depth_map, depth_diff_thres):
    if coord[0] < 0 or coord[1] < 0 or coord[0] >= check_map.shape[0] or coord[1] >= check_map.shape[1]:
        return
    if not check_map[tuple(coord)]:
        #if target_map[tuple(coord)] == 0 or target_map[tuple(coord)] == target_map_instance_id:
        if target_map[tuple(coord)] == 0:
            if check_keypoint_idx(coord, keypoint_idx, seg_map) and check_depth(coord, coord_prev, depth_map, depth_diff_thres):
                check_map[tuple(coord)] = 1

                target_map[tuple(coord)] = target_map_instance_id

                flood_fill(target_map, check_map, coord + np.array([0, 1]), coord, keypoint_idx, target_map_instance_id, seg_map, depth_map, depth_diff_thres)
                flood_fill(target_map, check_map, coord + np.array([1, 0]), coord, keypoint_idx, target_map_instance_id, seg_map, depth_map, depth_diff_thres)
                flood_fill(target_map, check_map, coord + np.array([0, -1]), coord, keypoint_idx, target_map_instance_id, seg_map, depth_map, depth_diff_thres)
                flood_fill(target_map, check_map, coord + np.array([-1, 0]), coord, keypoint_idx, target_map_instance_id, seg_map, depth_map, depth_diff_thres)

                check_map[tuple(coord)] = 2

def skeleton_depth(depth):
    def skeleton_depth_wrapper(keypoints):
        final_depth = 0
        keypoints_cnt =  0
        for coord in keypoints:
            coord = tuple(map(int, coord))
            coord = (coord[1], coord[0])
            if coord != (-1, -1):
                if depth[coord] >0:
                    final_depth += depth[coord]
                    keypoints_cnt += 1
        return float(final_depth) / keypoints_cnt
    return skeleton_depth_wrapper


def human_instance_seg(seg_map, depth_map, keypoints, depth_diff_thres=50):
    tar = np.zeros((seg_map.shape[0], seg_map.shape[1])).astype(np.uint8)
    check = np.zeros((seg_map.shape[0], seg_map.shape[1])).astype(np.uint8)
    keypoints = sorted(keypoints, key=skeleton_depth(depth_map))
    for idx, i in enumerate(keypoints):
        for j, coord in enumerate(i):
            coord = list(map(int, coord))
            #flood_fill(tar, check, np.array([coord[1], coord[0]]), np.array([coord[1], coord[0]]), j, idx + 1, seg_map, depth_map, depth_diff_thres)
            flood_fill(tar, np.zeros((seg_map.shape[0], seg_map.shape[1])).astype(np.uint8), np.array([coord[1], coord[0]]), np.array([coord[1], coord[0]]), j, idx + 1, seg_map, depth_map, depth_diff_thres)
    return tar