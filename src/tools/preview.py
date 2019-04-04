from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
import math
from PIL import Image
import argparse


def previewMulti(imgID):
    folders = sorted ( [i for i in glob ( './*' ) if osp.isdir ( i )] )

    rows = math.ceil ( math.sqrt ( len ( folders ) ) )

    for i, camName in enumerate ( folders ):
        imgs = sorted ( glob ( osp.join ( camName, '*.{jpg,png}' ) ) )
        curImg = Image.open ( imgs[imgID] )
        plt.subplot ( rows, rows, i + 1 )
        plt.imshow ( curImg )
        plt.xticks ( [] )
        plt.yticks ( [] )
        plt.title ( camName )

    plt.tight_layout ()

    plt.savefig ( './preview.png', dpi=100 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser ( description='Usage: python mat2pickle.py /parameter/dir /dir/to/dump' )
    parser.add_argument ( 'imgID', type=int )
    args = parser.parse_args ()
    previewMulti ( imgID=args.imgID )
