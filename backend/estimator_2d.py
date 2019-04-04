"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
"""
import sys
import os.path as osp

project_path = osp.abspath ( osp.join ( osp.dirname ( __file__ ), '..' ) )
if project_path not in sys.path:
    sys.path.insert ( 0, project_path )

from backend.light_head_rcnn.person_detector import PersonDetector
from backend.tf_cpn.Detector2D import Detector2D


class Estimator_2d ( object ):

    def __init__(self, DEBUGGING=False):
        self.bbox_detector = PersonDetector ( show_image=DEBUGGING )
        self.pose_detector_2d = Detector2D ( show_image=DEBUGGING )

    def estimate_2d(self, img, img_id):
        bbox_result = self.bbox_detector.detect ( img, img_id )
        dump_results = self.pose_detector_2d.detect ( bbox_result )
        return dump_results


if __name__ == '__main__':
    import cv2

    img = cv2.imread ( 'datasets/Shelf/Camera0/img_000000.png' )
    est = Estimator_2d ()
    est.estimate_2d ( img, 0 )
