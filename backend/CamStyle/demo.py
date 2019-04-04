"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
"""
from __future__ import print_function, absolute_import
import os.path as osp

from PIL import Image
import matplotlib.pyplot as plt

from .feature_extract import FeatureExtractor
from .reid.common_datasets import CommonDataset
import pdb


def demo(img_id):
    extractor = FeatureExtractor ()
    dataset = CommonDataset ( post_processed_dir='/home/jiangwen/Multi-Pose/backend/CamStyle/data/Shelf/post_processed' )
    data_batch = dataset[img_id]
    dismat = extractor.get_dismat ( data_batch, rerank=False )
    # query = list ( zip ( *data_batch[1:] ) )
    cnt = 0
    for i, cam_id in enumerate ( dataset.cam_names ):
        # Plot origin image
        base = 1
        info_dict = dataset.info_dict[cam_id][str ( img_id )]
        fname = info_dict.pop ( 'image_name', None )
        plt.subplot ( 5, 5, 5 * i + base )
        base += 1
        img = Image.open ( fname )
        plt.imshow ( img )
        plt.xlabel ( f'{osp.split(osp.split(fname)[-2])[-1]}/{osp.split(fname)[-1]}' )
        plt.xticks ( [] )
        plt.yticks ( [] )
        for k, v in info_dict.items ():
            plt.subplot ( 5, 5, 5 * i + base )
            # plot cropped image.
            base += 1
            img = Image.open ( v['img_path'] )
            plt.imshow ( img )
            fname = v['img_path']
            plt.xlabel ( f'{osp.split(osp.split(fname)[-2])[-1]}/{osp.split(fname)[-1]}#{cnt}' )
            cnt += 1
            plt.xticks ( [] )
            plt.yticks ( [] )
    plt.show ()
    return dismat


if __name__ == '__main__':
    img_id = 0
    demo ( img_id=img_id )
