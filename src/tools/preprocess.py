
import os
import sys
import os.path as osp
import pickle

project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )
import numpy as np
from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.m_utils.base_dataset import BaseDataset
from src.models.estimate3d import MultiEstimator


# from src.models.multiestimator import MultiEstimator\

def dump_mem(model, range_, loader, dump_dir):
    num_len = int ( np.log10 ( range_[-1] ) + 1 )
    for idx, imgs in enumerate ( tqdm ( loader ) ):
        # poses3d = model.estimate3d ( img_id=img_id, show=False )
        img_id = range_[idx]
        this_imgs = list ()
        for img_batch in imgs:
            this_imgs.append ( img_batch.squeeze ().numpy () )
        info_dicts = model._infer_single2d ( imgs=this_imgs )
        for cam_id, info_dict in info_dicts.items ():
            path2save = osp.join ( dump_dir, f"{img_id:0{num_len}d}.{cam_id}.image_data.npy" )

            np.save ( path2save, info_dict.pop ( 'image_data' ) )
            info_dict['image_path'] = osp.relpath ( path2save, dump_dir )
            for pid, person in enumerate ( info_dict[0] ):
                person['heatmap_path'] = osp.relpath (
                    osp.join ( dump_dir, f"{img_id:0{num_len}d}.{cam_id}.{pid}.heatmap_data.npy" ), dump_dir )
                np.save ( osp.join ( dump_dir, person['heatmap_path'] ),
                          person.pop ( 'heatmap_data' ) )
                person['cropped_path'] = osp.relpath (
                    osp.join ( dump_dir, f"{img_id:0{num_len}d}.{cam_id}.{pid}.cropped_img.npy" ), dump_dir )
                np.save ( osp.join ( dump_dir, person['cropped_path'] ), person.pop ( 'cropped_img' ) )

        with open ( osp.join ( dump_dir, f'{img_id:0{num_len}d}.info_dicts.pickle' ), 'wb' ) as f:
            pickle.dump ( info_dicts, f )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser (
        description='Usage: python preprocess.py -d Shelf Campus Panoptic [-dump_dir ./datasets]' )
    parser.add_argument ( '-d', nargs='+', dest='datasets' )
    parser.add_argument ( '-dump_dir', default=osp.join ( project_root, 'datasets' ), dest='dump_dir' )
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    # for template_mat in ['h36m', 'Shelf', 'Campus']:
    # for dataset_name in ['Panoptic']:
    for dataset_name in args.datasets:
        # for metric in ['geometry mean', 'Geometry only', 'ReID only']:
        model_cfg.testing_on = dataset_name
        # from backend.CamStyle.reid.common_datasets import load_template

        # template = load_template ( template_mat )
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            test_range = range ( 300, 600 )
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            test_range = [i for i in range ( 350, 471 )] + [i for i in range ( 650, 751 )]
            gt_path = dataset_path

        elif dataset_name == 'Panoptic':
            dataset_path = model_cfg.panoptic_ultimatum_path
            test_range = range ( 4500, 4900 )
            gt_path = osp.join ( dataset_path, '..' )
        elif dataset_name == 'ultimatum1':
            dataset_path = model_cfg.ultimatum1_path
            test_range = list ( range ( 10 * 25, 35 * 25, 10 ) ) + list ( range ( 60 * 25, 110 * 25, 10 ) ) + list (
                range ( 550 * 25, 600 * 25, 10 ) ) + list ( range ( 725 * 25, 770 * 25, 10 ) )
            # For 0min10s ~ 0min35s, 1min ~ 1min50s, 9min15s ~ 10min, 12min05s ~ 12min50s
        elif dataset_name == 'HD_ultimatum1':
            dataset_path = model_cfg.HD_ultimatum1_path
            test_range = model_cfg.HD_ultimatum1_range
        else:
            logger.error ( f"Unknown dataset name: {dataset_name}" )
            exit ( -1 )
        # print ( f'Using template on {template_mat}' )
        test_dataset = BaseDataset ( dataset_path, test_range )
        test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=12, shuffle=False )
        # test_dataset.template = template
        # test_model.dataset = test_dataset
        this_dump_dir = osp.join ( args.dump_dir, f'{dataset_name}_processed' )
        os.makedirs ( this_dump_dir, exist_ok=True )
        dump_mem ( test_model, test_range, test_loader, this_dump_dir )
