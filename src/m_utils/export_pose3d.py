import os
import os.path as osp
import pickle
import sys
import time

project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset


def export(model, loader, is_info_dicts=False, show=False):
    pose_list = list ()
    for img_id, imgs in enumerate ( tqdm ( loader ) ):
        try:
            pass
        except Exception as e:
            pass
            # poses3d = model.estimate3d ( img_id=img_id, show=False )
        if is_info_dicts:
            info_dicts = numpify ( imgs )

            model.dataset = MemDataset ( info_dict=info_dicts, camera_parameter=camera_parameter,
                                         template_name='Unified' )
            poses3d = model._estimate3d ( 0, show=show )
        else:
            this_imgs = list ()
            for img_batch in imgs:
                this_imgs.append ( img_batch.squeeze ().numpy () )
            poses3d = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                          show=show, plt_id=img_id )

        pose_list.append ( poses3d )
    return pose_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True,
                          choices=['Shelf', 'Campus', 'ultimatum1', 'HD_ultimatum1', 'HD_pizza', 'ultimatumVga10',
                                   'mafia2', '160224_mafia2'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            test_range = model_cfg.shelf_range
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            test_range = model_cfg.campus_range
            gt_path = dataset_path

        elif dataset_name == 'ultimatum1':
            dataset_path = model_cfg.ultimatum1_path
            test_range = model_cfg.ultimatum1_range
        elif dataset_name == 'ultimatumVga10':
            dataset_path = model_cfg.ultimatum1Vga10_path
            test_range = model_cfg.ultimatum1_range
        elif dataset_name == 'HD_ultimatum1':
            dataset_path = model_cfg.HD_ultimatum1_path
            test_range = model_cfg.HD_ultimatum1_range
        elif dataset_name == 'HD_pizza':
            dataset_path = model_cfg.pizza_path
            test_range = model_cfg.pizza_range
        elif dataset_name == 'mafia2':
            dataset_path = model_cfg.mafia2_demo_path
            test_range = model_cfg.mafia2_demo_range
        elif dataset_name == '160224_mafia2':
            dataset_path = model_cfg.mafia160244_path
            test_range = model_cfg.mafia160244_range
        else:
            logger.error ( f"Unknown datasets name: {dataset_name}" )
            exit ( -1 )

        # read the camera parameter of this dataset
        with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                    'rb' ) as f:
            camera_parameter = pickle.load ( f )

        # using preprocessed 2D poses or using CPN to predict 2D pose
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx] )
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
        else:
            test_dataset = BaseDataset ( dataset_path, test_range )

        test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        pose_in_range = export ( test_model, test_loader, is_info_dicts=bool ( args.dumped_dir ), show=True )
        with open ( osp.join ( model_cfg.root_dir, 'result',
                               time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M",
                                               time.localtime ( time.time () ) ) + '.pkl' ), 'wb' ) as f:
            pickle.dump ( pose_in_range, f )


