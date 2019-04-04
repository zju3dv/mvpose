
import numpy as np


def coco2shelf3D(coco_pose):
    """
    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param coco_pose: np.array with shape 3x17
    :return: 3D pose in shelf order with shape 14x3
    """
    coco_pose = coco_pose.T
    shelf_pose = np.zeros ( (14, 3) )
    coco2shelf = np.array ( [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9] )
    shelf_pose[0: 12] += coco_pose[coco2shelf]
    neck = (coco_pose[5] + coco_pose[6]) / 2  # L and R shoulder
    head_bottom = (neck + coco_pose[0]) / 2  # nose and head center
    head_center = (coco_pose[3] + coco_pose[4]) / 2  # middle of two ear
    # head_top = coco_pose[0] + (coco_pose[0] - head_bottom)
    head_top = head_bottom + (head_center - head_bottom) * 2
    # shelf_pose[12] += head_bottom
    # shelf_pose[13] += head_top
    shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2  # Use middle of shoulder to init
    shelf_pose[13] = coco_pose[0]  # use nose to init
    shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.75, 0.75, 1.5] )
    shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array ( [1. / 2., 1. / 2., 1. / 2.] )
    # shelf_pose[13] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [0.5, 0.5, 1.5] )
    # shelf_pose[12] = shelf_pose[12] + (shelf_pose[13] - shelf_pose[12]) * np.array ( [1.0 / 3, 1.0 / 3, 1.0 / 3] )
    return shelf_pose


def coco2panoptic(coco_pose):
    """

    :param coco_pose: 3x17 MS COCO17 order keypoints
    :return: 3x15 old style panoptic order keypoints
    """
    coco_pose = coco_pose.T
    panoptic_pose = np.zeros ( (15, 3) )
    map_array = np.array ( [5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16] )
    panoptic_pose[3:] += coco_pose[map_array]
    panoptic_pose[2] += (coco_pose[11] + coco_pose[12]) / 2  # Take middle of two hips as BodyCenter
    mid_shoulder = (coco_pose[5] + coco_pose[6]) / 2  # Use middle of shoulder to init
    nose = coco_pose[0]  # use nose to init
    head_top = mid_shoulder + (nose - mid_shoulder) * np.array ( [0.4, 1.75, 0.4] )
    neck = mid_shoulder + (nose - mid_shoulder) * np.array ( [.3, .5, .3] )
    panoptic_pose[0] += neck
    panoptic_pose[1] = head_top
    return panoptic_pose.T


def coco17to19(coco17pose):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    :param coco17pose: 17x3 coco pose np.array
    :return: 19x3 coco19 pose np.array
    """
    coco19pose = np.zeros ( (19, coco17pose.shape[1]) )
    index_array = np.array ( [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14] )
    coco19pose[index_array] = coco17pose
    coco19pose[0] = (coco17pose[5] + coco17pose[6]) / 2
    coco19pose[2] = (coco17pose[11] + coco17pose[12]) / 2
    coco19pose[-4:] = coco17pose[0]  # Since we have not implement eye and ear yet.
    return coco19pose
