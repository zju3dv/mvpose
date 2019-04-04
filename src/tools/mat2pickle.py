
import numpy as np
import scipy.io as scio
import pickle
import os.path as osp
import argparse


def mat2pickle(parameter_dir, dump_dir):
    K = np.stack ( scio.loadmat ( osp.join ( parameter_dir, 'intrinsic.mat' ) )['K'][0] ).astype ( np.float32 )
    P_mat = scio.loadmat ( osp.join ( parameter_dir, 'P.mat' ) )
    P = np.stack ( [np.stack ( p ) for p in P_mat['P'][0]] )
    RT = np.stack ( scio.loadmat ( osp.join ( parameter_dir, 'm_RT.mat' ) )['m_RT'][0] ).astype ( np.float32 )
    parameter_dict = {'K': K, 'P': P, 'RT': RT}

    with open ( osp.join ( dump_dir, 'camera_parameter.pickle' ), 'wb' ) as f:
        pickle.dump ( parameter_dict, f )


if __name__ == '__main__':
    parser = argparse.ArgumentParser ( description='Usage: python mat2pickle.py /parameter/dir /dir/to/dump' )
    parser.add_argument ( 'parameter_dir', type=str )
    parser.add_argument ( 'dump_dir', type=str )
    args = parser.parse_args ()
    mat2pickle ( parameter_dir=args.parameter_dir, dump_dir=args.dump_dir )
