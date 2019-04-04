
import os.path as osp
import sys

# Config project if not exist
project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..', '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )
# from src.m_utils.algorithm import transform_closure
from src.m_lib.pictorial import transform_closure
from src.models.match_solver import myproj2dpam
import time
import torch


def matchSVT(S, dimGroup, **kwargs):
    alpha = kwargs.get ( 'alpha', 0.1 )
    pSelect = kwargs.get ( 'pselect', 1 )
    tol = kwargs.get ( 'tol', 5e-4 )
    maxIter = kwargs.get ( 'maxIter', 500 )
    verbose = kwargs.get ( 'verbose', False )
    eigenvalues = kwargs.get ( 'eigenvalues', False )
    _lambda = kwargs.get ( '_lambda', 50 )
    mu = kwargs.get ( 'mu', 64 )
    dual_stochastic = kwargs.get ( 'dual_stochastic_SVT', True )
    if verbose:
        print ( 'Running SVT-Matching: alpha = %.2f, pSelect = %.2f _lambda = %.2f \n' % (
            alpha, pSelect, _lambda) )
    info = dict ()
    N = S.shape[0]
    S[torch.arange ( N ), torch.arange ( N )] = 0
    S = (S + S.t ()) / 2
    X = S.clone ()
    Y = torch.zeros_like ( S )
    W = alpha - S
    t0 = time.time ()
    for iter_ in range ( maxIter ):

        X0 = X
        # update Q with SVT
        U, s, V = torch.svd ( 1.0 / mu * Y + X )
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ diagS.diag () @ V.t ()
        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range ( len ( dimGroup ) - 1 ):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[torch.arange ( N ), torch.arange ( N )] = 1
        X[X < 0] = 0
        X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range ( len ( dimGroup ) - 1 ):
                row_begin, row_end = int ( dimGroup[i] ), int ( dimGroup[i + 1] )
                for j in range ( len ( dimGroup ) - 1 ):
                    col_begin, col_end = int ( dimGroup[j] ), int ( dimGroup[j + 1] )
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam ( X[row_begin:row_end, col_begin:col_end],
                                                                                1e-2 )

        X = (X + X.t ()) / 2
        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        pRes = torch.norm ( X - Q ) / N
        dRes = mu * torch.norm ( X - X0 ) / N
        if verbose:
            print ( f'Iter = {iter_}, Res = ({pRes}, {dRes}), mu = {mu}' )

        if pRes < tol and dRes < tol:
            break

        if pRes > 10 * dRes:
            mu = 2 * mu
        elif dRes > 10 * pRes:
            mu = mu / 2

    X = (X + X.t ()) / 2
    info['time'] = time.time () - t0
    info['iter'] = iter_

    if eigenvalues:
        info['eigenvalues'] = torch.eig ( X )

    X_bin = X > 0.5
    if verbose:
        print ( f"Alg terminated. Time = {info['time']}, #Iter = {info['iter']}, Res = ({pRes}, {dRes}), mu = {mu} \n" )
    match_mat = transform_closure ( X_bin.numpy() )
    return torch.tensor(match_mat)


if __name__ == '__main__':
    """
    Unit test, may only work on zjurv2.
    """
    import ipdb
    import pickle
    import os.path as osp
    import sys

    # Config project if not exist
    project_path = osp.abspath ( '.' )
    if project_path not in sys.path:
        sys.path.insert ( 0, project_path )
    from src.m_utils.visualize import show_panel_mem
    import numpy as np


    class TempDataset:
        def __init__(self, info_dict, cam_names):
            self.info_dicts = info_dict
            self.cam_names = cam_names

        def __getattr__(self, item):
            if item == 'info_dict':
                return self.info_dicts
            else:
                return self.cam_names


    with open ( '/home/jiangwen/Multi-Pose/result/0_match.pkl', 'rb' ) as f:
        d = pickle.load ( f )
    test_W = d[1][0].clone ()
    test_dimGroup = d[1][1]
    match_mat = matchSVT ( test_W, test_dimGroup, verbose=True )
    ipdb.set_trace ()
    bin_match = match_mat[:, torch.nonzero ( torch.sum ( match_mat, dim=0 ) > 1.9 ).squeeze ()] > 0.9
    bin_match = bin_match.reshape ( test_W.shape[0], -1 )

    matched_list = [[] for i in range ( bin_match.shape[1] )]
    for sub_imgid, row in enumerate ( bin_match ):
        if row.sum () != 0:
            pid = row.argmax ()
            matched_list[pid].append ( sub_imgid )

    matched_list = [np.array ( i ) for i in matched_list]
    info_batch = d[0]
    test_dataset = TempDataset ( info_batch[0], info_batch[1] )
    info_list = info_batch[3]
    sub_imgid2cam = info_batch[4]
    img_id = info_batch[5]
    affinity_mat = info_batch[6]
    geo_affinity_mat = info_batch[7]
    chosen_img = info_batch[-1]
    show_panel_mem ( test_dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat,
                     geo_affinity_mat, test_W, 0, [] )
