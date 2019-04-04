# encoding: utf-8
"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
This file is an middle-ware to encapsulate light-head-rcnn for academic use.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .light_config import cfg, config

from . import dataset
import os.path as osp
from . import network_desp
import tensorflow as tf
import numpy as np
import cv2, os, sys, math, json, pickle

from tqdm import tqdm
from utils.py_faster_rcnn_utils.cython_nms import nms, nms_new
from utils.py_utils import misc

from multiprocessing import Queue, Process
from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
from functools import partial



class PersonDetector ( object ):
    """
    This is an person detcor based on light-head-rcnn, which is adapted from test.py in experiments/.../ directory
    Some thing in light_config.py is hacked to simplify usage.
    Such as directory and output form.
    """

    def __init__(self, epoch_num=26, show_image=False):
        """
        Person detector init fuction, which will call tf.restore()
        The initialize process may cause many time with many tensorflow log printed out.
        :param epoch_num: 26 is the default model checkpoint
        :param show_image: show image to debug.
        """
        self._show_image = show_image
        self.model_file = osp.join (
            config.output_dir, 'model_dump',
            'epoch_{:d}'.format ( epoch_num ) + '.ckpt' )

        self.func, self.inputs = self._load_model ( self.model_file )

    def detect(self, image, image_id):
        """
        detect on a single image.
        :param image: full path of the image, not necessarily an absolute path
        :return: an dict of detection results.
        """
        all_results = []
        data_dict = dict ( data=image, image_id=image_id )
        result_dict = self.inference ( self.func, self.inputs, data_dict )
        all_results.append ( result_dict )

        if self._show_image:
            image = result_dict['data']
            for db in result_dict['result_boxes']:
                if db.score > config.test_vis_threshold:
                    db.draw ( image )
            if 'boxes' in result_dict.keys ():
                for db in result_dict['boxes']:
                    db.draw ( image )
            import matplotlib.pyplot as plt
            im = cv2.cvtColor ( image, cv2.COLOR_BGR2RGB )
            plt.imshow ( im )
            plt.show ()

        return self._make_result_dict ( all_results )

    def _load_model(self, model_file):
        from tensorflow.python import pywrap_tensorflow
        def get_variables_in_checkpoint_file(file_name):
            try:
                reader = pywrap_tensorflow.NewCheckpointReader ( file_name )
                var_to_shape_map = reader.get_variable_to_shape_map ()
                return var_to_shape_map
            except Exception as e:  # pylint: disable=broad-except
                print ( str ( e ) )
                if "corrupted compressed block contents" in str ( e ):
                    print (
                        "It's likely that your checkpoint file has been compressed "
                        "with SNAPPY." )

        def load_model(sess, model_path):
            # TODO(global variables ?? how about _adam weights)
            variables = tf.global_variables ()
            var_keep_dic = get_variables_in_checkpoint_file ( model_path )
            if 'global_step' in var_keep_dic:
                var_keep_dic.pop ( 'global_step' )

            # vis_var_keep_dic = []
            variables_to_restore = []
            for v in variables:
                if v.name.split ( ':' )[0] in var_keep_dic:
                    # print('Varibles restored: %s' % v.name)
                    variables_to_restore.append ( v )
                    # vis_var_keep_dic.append(v.name.split(':')[0])
                else:
                    # print('Unrestored Variables: %s' % v.name)
                    pass
            # print('Extra Variables in ckpt', set(var_keep_dic) - set(vis_var_keep_dic))

            if len ( variables_to_restore ) > 0:
                restorer = tf.train.Saver ( variables_to_restore )
                restorer.restore ( sess, model_path )
            else:
                print ( 'No variables in {} fits the network'.format ( model_path ) )

        tfconfig = tf.ConfigProto ( allow_soft_placement=True )
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session ( config=tfconfig )
        net = network_desp.Network ()
        inputs = net.get_inputs ()
        net.inference ( 'TEST', inputs )
        test_collect_dict = net.get_test_collection ()
        test_collect = [it for it in test_collect_dict.values ()]
        # The way in light-head is not work in my modified version since I rename variable names
        # saver = tf.train.Saver ()
        # saver.restore ( sess, model_file )
        # Use function in tf_cpn
        load_model ( sess, model_file )
        return partial ( sess.run, test_collect ), inputs

    def inference(self, val_func, inputs, data_dict):
        image = data_dict['data']
        ori_shape = image.shape

        if config.eval_resize == False:
            resized_img, scale = image, 1
        else:
            resized_img, scale = dataset.resize_img_by_short_and_max_size (
                image, config.eval_image_short_size, config.eval_image_max_size )
        height, width = resized_img.shape[0:2]

        resized_img = resized_img.astype ( np.float32 ) - config.image_mean
        resized_img = np.ascontiguousarray ( resized_img[:, :, [2, 1, 0]] )

        im_info = np.array (
            [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
            dtype=np.float32 )

        feed_dict = {inputs[0]: resized_img[None, :, :, :], inputs[1]: im_info}

        _, scores, pred_boxes, rois = val_func ( feed_dict=feed_dict )

        boxes = rois[:, 1:5] / scale

        if cfg.TEST.BBOX_REG:
            pred_boxes = bbox_transform_inv ( boxes, pred_boxes )
            pred_boxes = clip_boxes ( pred_boxes, ori_shape )

        pred_boxes = pred_boxes.reshape ( -1, config.num_classes, 4 )
        result_boxes = []
        for j in range ( 1, config.num_classes ):
            inds = np.where ( scores[:, j] > config.test_cls_threshold )[0]
            cls_scores = scores[inds, j]
            cls_bboxes = pred_boxes[inds, j, :]
            cls_dets = np.hstack ( (cls_bboxes, cls_scores[:, np.newaxis]) ).astype (
                np.float32, copy=False )

            keep = nms ( cls_dets, config.test_nms )
            cls_dets = np.array ( cls_dets[keep, :], dtype=np.float, copy=False )
            for i in range ( cls_dets.shape[0] ):
                db = cls_dets[i, :]
                dbox = DetBox (
                    db[0], db[1], db[2] - db[0], db[3] - db[1],
                    tag=config.class_names[j], score=db[-1] )
                result_boxes.append ( dbox )
        if len ( result_boxes ) > config.test_max_boxes_per_image:
            result_boxes = sorted (
                result_boxes, reverse=True, key=lambda t_res: t_res.score ) \
                [:config.test_max_boxes_per_image]

        result_dict = data_dict.copy ()
        result_dict['result_boxes'] = result_boxes
        return result_dict

    def _make_result_dict(self, all_results):
        coco_records = []

        for result in all_results:
            result_boxes = result['result_boxes']
            if config.test_save_type == 'coco':
                image_id = int ( result['image_id'] )
                for rb in result_boxes:
                    if rb.tag == 'person':
                        record = {'image_id': image_id, 'category_id': config.datadb.classes_originID[rb.tag],
                                  'score': rb.score, 'bbox': [rb.x, rb.y, rb.w, rb.h], 'data': result['data']}
                        coco_records.append ( record )
            else:
                raise Exception (
                    "Unimplemented save type: " + str ( config.test_save_type ) )

        return coco_records


if __name__ == '__main__':
    detector = PersonDetector ( show_image=True )
    img = cv2.imread ( './img_000000.png' )
    print ( detector.detect ( img, 0 ) )
