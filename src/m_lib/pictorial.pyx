#cython: boundscheck=False, wraparound=False, nonecheck=False
"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
"""
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
import scipy
import cv2
import cython
from cython.parallel import prange
from src.m_utils.geometry import check_bone_length
from libc.stdlib cimport malloc, free
from libc.string cimport memset
cdef int get_pa[13]
get_pa[:] = [-1, 0, 0, 1, 2, 3, 4, 0, 0, 7, 8, 9, 10]
cdef int edges2Joint[13]
from libc.math cimport exp as c_exp
from libc.math cimport sqrt as c_sqrt
edges2Joint[:] = [-1, 8, 9, 4, 5, 0, 1, 10, 11, 6, 7, 2, 3]

cpdef getskel():
    skel = {}
    skel['tree'] = [{} for i in range ( 13 )]
    skel['tree'][0]['name'] = 'Nose'
    skel['tree'][0]['children'] = [1, 2, 7, 8]
    skel['tree'][1]['name'] = 'LSho'
    skel['tree'][1]['children'] = [3]
    skel['tree'][2]['name'] = 'RSho'
    skel['tree'][2]['children'] = [4]
    skel['tree'][3]['name'] = 'LElb'
    skel['tree'][3]['children'] = [5]
    skel['tree'][4]['name'] = 'RElb'
    skel['tree'][4]['children'] = [6]
    skel['tree'][5]['name'] = 'LWri'
    skel['tree'][5]['children'] = []
    skel['tree'][6]['name'] = 'RWri'
    skel['tree'][6]['children'] = []
    skel['tree'][7]['name'] = 'LHip'
    skel['tree'][7]['children'] = [9]
    skel['tree'][8]['name'] = 'RHip'
    skel['tree'][8]['children'] = [10]
    skel['tree'][9]['name'] = 'LKne'
    skel['tree'][9]['children'] = [11]
    skel['tree'][10]['name'] = 'RKne'
    skel['tree'][10]['children'] = [12]
    skel['tree'][11]['name'] = 'LAnk'
    skel['tree'][11]['children'] = []
    skel['tree'][12]['name'] = 'RAnk'
    skel['tree'][12]['children'] = []
    return skel

cdef struct Edge:
    int child
    int parent
    double bone_mean
    double bone_std

cdef Edge* getPictoStruct(skel, distribution):
    """to get the pictorial structure"""
    graph = skel['tree']
    level = np.zeros ( len ( graph ) )
    for i in range ( len ( graph ) ):
        queue = np.array ( graph[i]['children'], dtype=np.int32 )
        for j in range ( queue.shape[0] ):
            graph[queue[j]]['parent'] = i
        while queue.shape[0] != 0:
            level[queue[0]] = level[queue[0]] + 1
            queue = np.append ( queue, graph[queue[0]]['children'] )
            queue = np.delete ( queue, 0 )
            queue = np.array ( queue, dtype=np.int32 )
    trans_order = np.argsort ( -level )
    # edges = [{} for i in range ( len ( trans_order ) - 1 )]
    cdef Edge* edges = <Edge*>malloc(sizeof(Edge)*( len ( trans_order ) - 1 ))
    for i in range ( len ( trans_order )-1):
        # print(edges[i])
        edges[i].child = trans_order[i]
        edges[i].parent = graph[edges[i].child]['parent']
        edge_id = distribution['joints2edges'][(edges[i].child, edges[i].parent)]
        edges[i].bone_mean = distribution['mean'][edge_id]
        edges[i].bone_std = distribution['std'][edge_id]
        # print(edges[i])

    return edges

cdef double get_prior(int i, int n, int p, int j, Edge* edges, np.ndarray[np.float64_t, ndim=3]X):
    """calculate the probability p(si,sj)"""
    cdef int edges_2_joint[13]
    edges_2_joint[:] = [-1, 8, 9, 4, 5, 0, 1, 10, 11, 6, 7, 2, 3]
    bone_std = edges[edges_2_joint[i]].bone_std
    bone_mean = edges[edges_2_joint[i]].bone_mean
    distance = np.linalg.norm ( X[i][n] - X[p][j] )
    relative_error = np.abs ( distance - bone_mean ) / bone_std
    prior = scipy.stats.norm.sf ( relative_error ) * 2
    return prior

cdef  get_max(int i, int p, int j, np.ndarray[np.float64_t, ndim=2] unary, int candidateNum, Edge* edges,
                    np.ndarray[np.float64_t, ndim=3] X):
    # i : joint index, p : i's parent joint, j : p's jth point
    unary_sum = np.zeros ( candidateNum )
    for n in range ( candidateNum ):
        prior = get_prior ( i, n, p, j, edges, X )
        unary_sum[n] = prior + unary[i][n]
    this_max = np.max ( unary_sum )
    index = np.where ( unary_sum == np.max ( unary_sum ) )[0][0]
    return this_max, index

@cython.cdivision(True)
cdef inferPict3D_MaxProd(np.ndarray[np.float64_t, ndim=2, mode="c"]unary, Edge* edges, np.ndarray[np.float64_t, ndim=3, mode="c"]X):
    """
    To inference the pictorial structure in parallel
    """
    cdef double[:,:]unary_c = unary
    cdef double[:, :, :] X_c = X
    cdef int jointNum = unary.shape[0]
    cdef int p
    cdef double m
    cdef int candidateNum = unary[0].shape[0]
    cdef int curJoint, parentCandidate, curCandidate
    cdef int maxInArray
    # cdef double [:,:]unary_sum
    cdef double bone_mean, bone_std, prior, distance
    cdef double maxUnary
    for curJoint in range ( jointNum - 1, 0, -1 ):
        parentJoint = get_pa[curJoint]
        for parentCandidate in prange ( candidateNum, nogil=True):
            maxUnary = -100000 # very negative value
            for curCandidate in range ( candidateNum ):
                # Begin of get prior
                bone_std = edges[edges2Joint[curJoint]].bone_std
                bone_mean = edges[edges2Joint[curJoint]].bone_mean
                distance = c_sqrt((X_c[curJoint][curCandidate][0] - X_c[parentJoint][parentCandidate][0])**2+
                                  (X_c[curJoint][curCandidate][1] - X_c[parentJoint][parentCandidate][1])**2+
                                  (X_c[curJoint][curCandidate][2] - X_c[parentJoint][parentCandidate][2])**2)
                # relative_error = (distance - bone_mean) / bone_std
                prior = c_exp(-(distance-bone_mean)**2/(2*bone_std**2))/bone_std
                # end of get prior
                if prior + unary_c[curJoint][curCandidate] > maxUnary:
                    maxUnary = prior + unary_c[curJoint][curCandidate]
            unary_c[parentJoint][parentCandidate] += maxUnary
    # get the max index

    values = unary[0]
    xpk = np.zeros ( unary.shape[0], dtype=np.int64 )  # Also change from original implementation
    xpk[0] = values.argmax ()
    for curJoint in range ( 1, jointNum ):
        parentJoint = get_pa[curJoint]
        xn = get_max ( curJoint, parentJoint, xpk[parentJoint], unary, candidateNum, edges, X )
        xpk[curJoint] = xn[1]
    return xpk

def hybrid_kernel(model, matched_list, pose_mat, sub_imgid2cam, img_id):
    multi_pose3d = list ()
    for person in matched_list:
        # use bottom-up approach to get the 3D pose of person
        if person.shape[0] <= 1:
            continue

        # step1: use the 2D joint of person to triangulate the 3D joints candidates

        # person's 17 3D joints candidates
        candidates = np.zeros ( (17, person.shape[0] * (person.shape[0] - 1) // 2, 3) )
        # 17xC^2_nx3
        cnt = 0
        for i in range ( person.shape[0] ):
            for j in range ( i + 1, person.shape[0] ):
                cam_id_i, cam_id_j = sub_imgid2cam[person[i]], sub_imgid2cam[person[j]]
                projmat_i, projmat_j = model.dataset.P[cam_id_i], model.dataset.P[cam_id_j]
                pose2d_i, pose2d_j = pose_mat[person[i]].T, pose_mat[person[j]].T
                pose3d_homo = cv2.triangulatePoints ( projmat_i, projmat_j, pose2d_i, pose2d_j )
                pose3d_ij = pose3d_homo[:3] / pose3d_homo[3]
                candidates[:, cnt] += pose3d_ij.T
                cnt += 1

        unary = model.dataset.get_unary ( person, sub_imgid2cam, candidates, img_id )

        # step2: use the max-product algorithm to inference to get the 3d joint of the person

        # change the coco order
        coco_2_skel = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        candidates = np.array ( candidates )[coco_2_skel]
        unary = unary[coco_2_skel]
        skel = getskel ()
        # construct pictorial model
        edges = getPictoStruct ( skel, model.dataset.distribution )
        # print(f'unary: {type(candidates)}')
        # print(f'edges: {type(edges))}')
        # print(f'candidates: {type(candidates)}')
        xp = inferPict3D_MaxProd ( unary, edges, candidates )
        human = np.array ( [candidates[i][j] for i, j in zip ( range ( xp.shape[0] ), xp )] )
        human_coco = np.zeros ( (17, 3) )
        human_coco[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = human
        human_coco[[1, 2, 3, 4]] = human_coco[0]  # Just make visualize beauty not real ear and eye
        human_coco = human_coco.T
        if check_bone_length ( human_coco ):
            multi_pose3d.append ( human_coco )
        free(edges)
    return multi_pose3d

cpdef transform_closure(np.ndarray[np.uint8_t, ndim=2, mode="c"] X_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param X_bin: torch.tensor which is binarized by a threshold
    :return:
    """
    # temp = np.zeros_like ( X_bin )
    cdef int N = X_bin.shape[0]
    # cdef np.uint8_t[:,:] X_bin_c = X_bin
    cdef int *temp = <int*>malloc(N*N*sizeof(int))
    memset(temp,0, sizeof(int)*N*N)
    # temp[...] = 0
    # temp = cvarray(shape=(N, N), itemsize=sizeof(int), format="i")
    cdef int i, j, k
    for k in range ( N ):
        for i in range ( N ):
            for j in range ( N ):
                # temp[i][j] = X_bin_c[i, j] or (X_bin_c[i, k] and X_bin_c[k, j])
                temp[i*N+j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
    # vis = cvarray(shape=(N), itemsize=sizeof(int), format="i")
    # vis[...] = 0
    cdef int* vis = <int*>malloc(N*sizeof(int))
    memset(vis, 0, sizeof(int)*N)
    match_mat = np.zeros_like ( X_bin )
    for i in range(N):
        if vis[i]:
            continue
        for j in range(N):
            if temp[i*N+j]:
                vis[j] = 1
                match_mat[j, i] = 1
    free(temp)
    free(vis)
    return match_mat
