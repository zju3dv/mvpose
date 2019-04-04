import numpy as np
import scipy





def getskel():
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


def getPictoStruct(skel, distribution):
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
    edges = [{} for i in range ( len ( trans_order ) - 1 )]
    for i in range ( len ( trans_order ) - 1 ):
        edges[i]['child'] = trans_order[i]
        edges[i]['parent'] = graph[edges[i]['child']]['parent']
        edge_id = distribution['joints2edges'][(edges[i]['child'], edges[i]['parent'])]
        edges[i]['bone_mean'] = distribution['mean'][edge_id]
        edges[i]['bone_std'] = distribution['std'][edge_id]
    return edges


def get_prior(i, n, p, j, edges, X):
    """calculate the probability p(si,sj)"""
    import math

    edges_2_joint = [[], 8, 9, 4, 5, 0, 1, 10, 11, 6, 7, 2, 3]
    bone_std = edges[edges_2_joint[i]]['bone_std']
    bone_mean = edges[edges_2_joint[i]]['bone_mean']
    distance = np.linalg.norm ( X[i][n] - X[p][j] )
    # TODO: Change to gaussian distribution
    relative_error = np.abs ( distance - bone_mean ) / bone_std
    prior = scipy.stats.norm.sf ( relative_error ) * 2
    return prior


def get_max(i, p, j, unary, edges, X):
    # i : joint index, p : i's parent joint, j : p's jth point
    # unary_sum = np.array ( [0 for n in range ( len ( unary[i] ) )] ) # Change from original implementation
    # import ipdb
    # ipdb.set_trace()
    unary_sum = np.zeros ( len ( unary[i] ) )
    for n in range ( len ( unary[i] ) ):
        prior = get_prior ( i, n, p, j, edges, X )
        unary_sum[n] = prior + unary[i][n]
    this_max = np.max ( unary_sum )
    index = np.where ( unary_sum == np.max ( unary_sum ) )[0][0]
    return this_max, index


def get_pa(i):
    child = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    parent = [[], 0, 0, 1, 2, 3, 4, 0, 0, 7, 8, 9, 10]

    return parent[child[i]]


def inferPict3D_MaxProd(unary, edges, X):
    """to inference the pictorial structure"""

    num = unary.shape[0]
    for i in range ( num - 1, 0, -1 ):
        p = get_pa ( i )
        for j in range ( unary[p].shape[0] ):
            m = get_max ( i, p, j, unary, edges, X )
            unary[p][j] = unary[p][j] + m[0]
    # get the max index

    values = unary[0]
    # xpk = np.array ( [0 for i in range ( unary.shape[0] )] )
    xpk = np.zeros ( unary.shape[0], dtype=np.int64 )  # Also change from original implementation
    xpk[0] = values.argmax ()
    import ipdb
    ipdb.set_trace()
    for n in range ( 1, num ):
        p = get_pa ( n )
        xn = get_max ( n, p, xpk[p], unary, edges, X )
        xpk[n] = xn[1]
    return xpk
