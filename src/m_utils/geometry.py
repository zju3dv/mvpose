
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim



def projected_distance_origin(pts_0, pts_1, F):
    """
    Compute point distance with epipolar geometry knowledge
    :param pts_0: numpy points array with shape Nx2
    :param pts_1: numpy points array with shape Nx2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """
    lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2,
                                            F )  # I know 2 is not seems right, but it actually work for this dataset
    lines = lines.reshape ( -1, 3 )
    points_1 = np.ones ( (pts_0.shape[0], 3) )
    points_1[:, :2] = pts_1
    dist = np.sum ( lines * points_1, axis=1 ) / np.linalg.norm ( lines[:, :2], axis=1 )
    dist = np.abs ( dist )
    dist = np.mean ( dist )
    return dist


def projected_distance(pts_0, pts_1, F):
    """
    Compute point distance with epipolar geometry knowledge
    :param pts_0: numpy points array with shape Nx17x2
    :param pts_1: numpy points array with shape Nx17x2
    :param F: Fundamental matrix F_{01}
    :return: numpy array of pairwise distance
    """
    # lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2,
    #                                         F )  # I know 2 is not seems right, but it actually work for this dataset
    # lines = lines.reshape ( -1, 3 )
    # points_1 = np.ones ( (lines.shape[0], 3) )
    # points_1[:, :2] = pts_1.reshape((-1, 2))
    #
    # # to begin here!
    # dist = np.sum ( lines * points_1, axis=1 ) / np.linalg.norm ( lines[:, :2], axis=1 )
    # dist = np.abs ( dist )
    # dist = np.mean ( dist )


    lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2, F )
    lines = lines.reshape(-1, 17, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], 17, 3])
    points_1[0, :, :, :2] = pts_1

    dist = np.sum(lines * points_1, axis=3) #/ np.linalg.norm(lines[:, :, :, :2], axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)



    return dist


def geometry_affinity(points_set, Fs, dimGroup):
    M, _, _ = points_set.shape
    # distance_matrix = np.zeros ( (M, M), dtype=np.float32 )
    distance_matrix = np.ones ( (M, M), dtype=np.float32 ) * 25
    np.fill_diagonal(distance_matrix, 0)
    # TODO: remove this stupid nested for loop
    import time
    start_time = time.time()
    for cam_id0, h in enumerate ( range ( len ( dimGroup ) - 1 ) ):
        for cam_add, k in enumerate ( range ( cam_id0+1, len(dimGroup)-1 ) ):
            cam_id1 = cam_id0 + cam_add + 1
            # if there is no one in some view, skip it!
            if dimGroup[h] == dimGroup[h+1] or dimGroup[k] == dimGroup[k+1]:
                continue

            pose_id0 = points_set[dimGroup[h]:dimGroup[h + 1]]
            pose_id1 = points_set[dimGroup[k]:dimGroup[k + 1]]
            distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]] = \
                (projected_distance(pose_id0, pose_id1, Fs[cam_id0, cam_id1]) + \
                 projected_distance(pose_id1, pose_id0, Fs[cam_id1, cam_id0]).T) / 2
            distance_matrix[dimGroup[k]:dimGroup[k+1], dimGroup[h]:dimGroup[h+1]] = \
                distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]].T

    end_time = time.time()
    # print('using %fs' % (end_time - start_time))
    if distance_matrix.std() < 5:
        for i in range(distance_matrix.shape[0]):
            distance_matrix[i, i] = distance_matrix.mean()

    affinity_matrix = - (distance_matrix - distance_matrix.mean ()) / distance_matrix.std ()
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp ( -5 * affinity_matrix ))
    return affinity_matrix

def geometry_affinity_origin(points_set, Fs, dimGroup):
    M, _, _ = points_set.shape
    distance_matrix = np.zeros ( (M, M), dtype=np.float32 )
    # TODO: remove this stupid nested for loop
    for cam_id0, h in enumerate ( range ( len ( dimGroup ) - 1 ) ):
        for cam_id1, k in enumerate ( range ( len ( dimGroup ) - 1 ) ):
            for i in range ( dimGroup[h], dimGroup[h + 1] ):
                for j in range ( dimGroup[k], dimGroup[k + 1] ):
                    distance_matrix[i, j] += (projected_distance ( points_set[i], points_set[j],
                                                                   Fs[cam_id0, cam_id1] ) + projected_distance (
                        points_set[j], points_set[i], Fs[cam_id1, cam_id0] )) / 2
    affinity_matrix = - (distance_matrix - distance_matrix.mean ()) / distance_matrix.std ()
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp ( -5 * affinity_matrix ))
    return affinity_matrix


def get_min_reprojection_error(person, dataset, pose_mat, sub_imgid2cam):
    reproj_error = np.zeros ( (len ( person ), len ( person )) )
    for i, p0 in enumerate ( person ):
        for j, p1 in enumerate ( person ):
            projmat_0 = dataset.P[sub_imgid2cam[p0]]
            projmat_1 = dataset.P[sub_imgid2cam[p1]]
            pose2d_0, pose2d_1 = pose_mat[p0].T, pose_mat[p1].T
            pose3d_homo = cv2.triangulatePoints ( projmat_0, projmat_1, pose2d_0, pose2d_1 )
            this_error = 0
            for pk in person:
                projmat_k = dataset.P[sub_imgid2cam[pk]]
                projected_pose_k_homo = projmat_k @ pose3d_homo
                projected_pose_k = projected_pose_k_homo[:2] / projected_pose_k_homo[2]
                this_error += np.linalg.norm ( projected_pose_k - pose_mat[pk].T )
            reproj_error[i, j] = this_error

    reproj_error[np.arange ( len ( person ) ), np.arange ( len ( person ) )] = np.inf
    # TODO: figure out why NaN
    reproj_error[np.isnan ( reproj_error )] = np.inf
    x, y = np.where ( reproj_error == reproj_error.min () )
    sub_imageid = np.array ( [person[x[0]], person[y[0]]] )
    return sub_imageid


def check_bone_length(pose_3d):
    """
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    :param pose_3d: 3xN 3D pose in MSCOCO order
    :return: boolean
    """
    min_length = 0.1
    max_length = 0.7
    _BONES = [[5, 7], [6, 8], [7, 9], [8, 10], [11, 13], [12, 14], [13, 15], [14, 16]]
    error_cnt = 0
    for kp_0, kp_1 in _BONES:
        bone_length = np.sqrt ( np.sum ( (pose_3d[:, kp_0] - pose_3d[:, kp_1]) ** 2 ) )
        if bone_length < min_length or bone_length > max_length:
            error_cnt += 1

    return error_cnt < 3


def bundle_adjustment(pose3d_homo, person, dataset, pose_mat, sub_imgid2cam, bundle=50, logging=None):
    dtype = torch.float32
    temp_pose = nn.Parameter ( torch.tensor ( pose3d_homo.copy (), dtype=dtype ) )
    optimizer = optim.Adam ( [temp_pose] )
    loss_summary = list ()
    for it in range ( 5000 ):
        optimizer.zero_grad ()
        loss = torch.tensor ( 0. )
        for p in person:
            projmat = torch.tensor ( dataset.P[sub_imgid2cam[p]], dtype=dtype )
            projected_pose = projmat @ temp_pose
            projected_pose = (projected_pose[:2] / projected_pose[2]).t ()  # 2x17 -> 17x2 tensor
            raw_pose = torch.tensor ( pose_mat[p], dtype=dtype )  # 17x2 tensor
            # TODO: make joint wise pose adjustment
            this_loss = torch.sum ( torch.sqrt ( (raw_pose - projected_pose) ** 2 ), dim=1 )
            this_loss[(this_loss > bundle) + torch.isnan ( this_loss )] *= 0
            loss += torch.sum ( this_loss )
            logging.info ( f"{it} iter, p_{p}: {float(torch.sum(this_loss))} px\n\t {this_loss}" )
            # if this_loss < bundle and not torch.isnan ( this_loss ):  # torch.sqrt(0) is nan.
            #     loss += this_loss
            # else:
            #     pass
        if loss < 1e-1 or torch.isnan ( loss ) or (it > 10 and abs ( loss - loss_summary[-1] ) < 0.1):
            #     if loss < 1e-1 or torch.isnan ( loss ):
            break
        loss.backward ()
        optimizer.step ()
        # print ( temp_pose[:, 12] )
        loss_summary.append ( float ( loss ) )
    logging.info ( f"{it} iter, {loss} px" )
    pose3d_homo = torch.tensor ( temp_pose, dtype=dtype )
    return pose3d_homo.clone ().detach ().numpy ()


def multiTri(Ps, Ys):
    """

    :param Ps: Nx3x4 Projection matrix
    :param Ys: Nx2 Correspond 2D keypoints
    :return: Xs: Nx3 3D keypoints
    """
    Ys_homo = torch.ones ( (Ys.shape[0], 3, 1) )
    Ys_homo[:, :2] = Ys.reshape ( [-1, 2, 1] )

    Xs_homo = torch.sum ( Ps.transpose ( 2, 1 ) @ Ps, dim=0 ).inverse () @ torch.sum ( Ps.transpose ( 2, 1 ) @ Ys_homo,
                                                                                       dim=0 )

    Xs = Xs_homo[:3] / Xs_homo[3]
    return Xs


def multiTriIter(Ps, Ys, lr=1e-3):
    """

    :param Ps: torch.tensor of Projection matrix
    :param Ys: torch.tensor of Nx2xJ  2D keypoints
    :param lr: step size for Adam optimizer
    :return:
    """
    N, _, J = Ys.shape
    initPose = cv2.triangulatePoints ( Ps[0].numpy (), Ps[1].numpy (), Ys[0].numpy (), Ys[1].numpy () )
    gdPose = nn.Parameter ( torch.tensor ( initPose, dtype=torch.float32 ) )

    optimizer = optim.Adam ( [gdPose], lr=lr )
    last_loss = torch.tensor ( 0. )
    for step in range ( 1000 ):
        optimizer.zero_grad ()
        loss = torch.tensor ( 0. )
        for i, P in enumerate ( Ps ):
            projected_XsHomo = P @ gdPose
            projected_Xs = projected_XsHomo[:2] / projected_XsHomo[2]
            loss += torch.norm ( projected_Xs - Ys[i] )

        if torch.abs ( last_loss - loss ) < 1e-3:
            break
        else:
            last_loss = loss
        loss.backward ()
        optimizer.step ()
    return gdPose.detach ()
