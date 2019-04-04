"""
@author: Jiang Wen
@contact: Wenjiang.wj@foxmail.com
This file is adapted from tf_CPN
"""
import os
import os.path as osp
import numpy as np
import argparse
from .cpn_config import cfg
import cv2
import sys
import time
import scipy
import copy
import tensorflow as tf

from tfflat.base import Tester
from .network import Network

from lib_nms.gpu_nms import gpu_nms
from lib_nms.cpu_nms import cpu_soft_nms
from dataset import Preprocessing


class Detector2D ( object ):

    def __init__(self, test_model=os.path.join ( cfg.cur_dir, 'log/model_dump/snapshot_350.ckpt' ), show_image=False):
        self.show_image = show_image
        self.tester = Tester ( Network (), cfg )
        self.tester.load_weights ( test_model )

    def detect(self, dets):
        dets = [i for i in dets if i['category_id'] == 1]
        dets.sort ( key=lambda x: (x['image_id'], x['score']), reverse=True )
        dump_results = self._test_net ( dets )
        return dump_results

    def _test_net(self, dets):
        # here we assume all boxes are pre-processed.
        det_range = [0, len ( dets )]
        nms_method = 'nms'
        nms_thresh = 1.
        min_scores = 0.5  # 1e-10 modified to avoid mismatch
        min_box_size = 0.  # 8 ** 2

        all_res = []
        dump_results = []

        start_time = time.time ()

        img_start = det_range[0]
        while img_start < det_range[1]:
            img_end = img_start + 1
            im_info = dets[img_start]
            while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
                img_end += 1

            test_data = dets[img_start:img_end]
            img_start = img_end
            all_res.append ( [] )
            # get box detections
            cls_dets = np.zeros ( (len ( test_data ), 5), dtype=np.float32 )
            for i in range ( len ( test_data ) ):
                bbox = np.asarray ( test_data[i]['bbox'] )
                cls_dets[i, :4] = np.array ( [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] )
                cls_dets[i, 4] = np.array ( test_data[i]['score'] )

            # nms and filter
            keep = np.where ( (cls_dets[:, 4] >= min_scores) &
                              ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size) )[
                0]
            cls_dets = cls_dets[keep]
            if len ( cls_dets ) > 0:
                if nms_method == 'nms':
                    keep = gpu_nms ( cls_dets, nms_thresh )
                elif nms_method == 'soft':
                    keep = cpu_soft_nms ( np.ascontiguousarray ( cls_dets, dtype=np.float32 ), method=2 )
                else:
                    assert False
            cls_dets = cls_dets[keep]
            test_data = np.asarray ( test_data )[keep]

            if len ( keep ) == 0:
                continue

            # crop and detect keypoints
            cls_skeleton = np.zeros ( (len ( test_data ), cfg.nr_skeleton, 3) )
            crops = np.zeros ( (len ( test_data ), 4) )
            cfg.batch_size = 32
            batch_size = cfg.batch_size // 2
            for test_id in range ( 0, len ( test_data ), batch_size ):
                start_id = test_id
                end_id = min ( len ( test_data ), test_id + batch_size )

                test_imgs = []
                details = []
                for i in range ( start_id, end_id ):
                    test_img, detail = Preprocessing ( test_data[i], stage='test' )
                    test_imgs.append ( test_img )
                    details.append ( detail )

                details = np.asarray ( details )
                feed = test_imgs
                for i in range ( end_id - start_id ):
                    ori_img = test_imgs[i][0].transpose ( 1, 2, 0 )
                    flip_img = cv2.flip ( ori_img, 1 )
                    feed.append ( flip_img.transpose ( 2, 0, 1 )[np.newaxis, ...] )
                feed = np.vstack ( feed )

                res = self.tester.predict_one ( [feed.transpose ( 0, 2, 3, 1 ).astype ( np.float32 )] )[0]
                # from IPython import embed; embed()
                res = res.transpose ( 0, 3, 1, 2 )

                for i in range ( end_id - start_id ):
                    fmp = res[end_id - start_id + i].transpose ( (1, 2, 0) )
                    fmp = cv2.flip ( fmp, 1 )
                    fmp = list ( fmp.transpose ( (2, 0, 1) ) )
                    for (q, w) in cfg.symmetry:
                        fmp[q], fmp[w] = fmp[w], fmp[q]
                    fmp = np.array ( fmp )
                    res[i] += fmp
                    res[i] /= 2

                heatmaps = []
                for test_image_id in range ( start_id, end_id ):
                    r0 = res[test_image_id - start_id].copy ()
                    r0 /= 255.
                    r0 += 0.5
                    for w in range ( cfg.nr_skeleton ):
                        res[test_image_id - start_id, w] /= np.amax ( res[test_image_id - start_id, w] )
                    border = 10
                    dr = np.zeros (
                        (cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border) )
                    dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy ()
                    # TODO: try to use those with out gaussian
                    for w in range ( cfg.nr_skeleton ):
                        dr[w] = cv2.GaussianBlur ( dr[w], (21, 21), 0 )
                        # dr[w] = cv2.GaussianBlur ( dr[w], (1, 1), 0 ) # Will working on it.
                    raw_heatmaps = list ( dr[:, border:-border, border:-border].copy () )
                    for w in range ( cfg.nr_skeleton ):
                        lb = dr[w].argmax ()
                        y, x = np.unravel_index ( lb, dr[w].shape )
                        dr[w, y, x] = 0
                        lb = dr[w].argmax ()
                        py, px = np.unravel_index ( lb, dr[w].shape )
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max ( 0, min ( x, cfg.output_shape[1] - 1 ) )
                        y = max ( 0, min ( y, cfg.output_shape[0] - 1 ) )
                        cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
                        cls_skeleton[test_image_id, w, 2] = r0[
                            w, int ( round ( y ) + 1e-10 ), int ( round ( x ) + 1e-10 )]
                    # map back to original images
                    crops[test_image_id, :] = details[test_image_id - start_id, :]
                    length = crops[test_image_id][2] - crops[test_image_id][0]
                    width = crops[test_image_id][3] - crops[test_image_id][1]
                    l_ori = raw_heatmaps[0].shape[0]
                    w_ori = raw_heatmaps[0].shape[1]

                    def test_for_loop(raw_heatmaps, order=3):
                        heatmaps_this = copy.deepcopy ( raw_heatmaps )
                        for w in range ( cfg.nr_skeleton ):
                            heatmaps_this[w] = scipy.ndimage.interpolation.zoom ( raw_heatmaps[w],
                                                                                  (width / l_ori, length / w_ori),
                                                                                  order=order )
                            heatmaps_this[w][heatmaps_this[w] <= 0] = 10e-6
                        zoomed_heatmaps = np.array ( heatmaps_this )
                        return zoomed_heatmaps

                    def test_my(raw_heatmaps, order=3):
                        origin_heatmap = np.array ( raw_heatmaps )
                        zoomed_heatmaps = scipy.ndimage.interpolation.zoom ( origin_heatmap,
                                                                             (1, width / l_ori, length / w_ori),
                                                                             order=order )
                        zoomed_heatmaps[zoomed_heatmaps <= 0] = 10e-6
                        return zoomed_heatmaps

                    # origin_heatmap = np.array ( raw_heatmaps )
                    zoomed_heatmaps = np.empty ( (len ( raw_heatmaps ), int ( width ), int ( length )) )
                    for zoom_id, heatmap_i in enumerate ( raw_heatmaps ):
                        zoomed_heatmaps[zoom_id] = scipy.ndimage.interpolation.zoom ( heatmap_i,
                                                                                      (width / l_ori,
                                                                                       length / w_ori),
                                                                                      order=3 )
                    # zoomed_heatmaps = np.array ( raw_heatmaps )
                    zoomed_heatmaps[zoomed_heatmaps <= 0] = 10e-6
                    # orinial_heatmaps = np.zeros((cfg.nr_skeleton, ori_width, ori_length))
                    # orinial_heatmaps[:, int(crops[test_image_id][1]):int(crops[test_image_id][3]), int(crops[test_image_id][0]):int(crops[test_image_id][2])] = heatmaps_this
                    # orinial_heatmaps[orinial_heatmaps <= 0] = 10**-6
                    heatmaps.append ( zoomed_heatmaps )
                    for w in range ( cfg.nr_skeleton ):
                        cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[1] * (
                                crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
                        cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[0] * (
                                crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
            all_res[-1] = [cls_skeleton.copy (), cls_dets.copy ()]

            cls_partsco = cls_skeleton[:, :, 2].copy ().reshape ( -1, cfg.nr_skeleton )
            cls_skeleton[:, :, 2] = 1
            cls_scores = cls_dets[:, -1].copy ()

            # rescore
            cls_dets[:, -1] = cls_scores * cls_partsco.mean ( axis=1 )
            cls_skeleton = np.concatenate (
                [cls_skeleton.reshape ( -1, cfg.nr_skeleton * 3 ),
                 (cls_scores * cls_partsco.mean ( axis=1 ))[:, np.newaxis]],
                axis=1 )
            for i in range ( len ( cls_skeleton ) ):
                result = dict ( image_id=im_info['image_id'], category_id=1,
                                score=float ( round ( cls_skeleton[i][-1], 4 ) ),
                                keypoints=cls_skeleton[i][:-1].round ( 3 ).tolist (), bbox=dets[i]['bbox'],
                                heatmaps=heatmaps[i], crops=crops[i] )
                dump_results.append ( result )
            if self.show_image:
                import pdb;
                pdb.set_trace ()
                dbg_im = dets[0]['data']  # Since all detection are based on the same image
                from utils.visualize import visualize
                visualize ( dbg_im, keypoints=[i['keypoints'] for i in dump_results], det_boxes=cls_dets )
                # import pdb; pdb.set_trace ()
        # return all_res, dump_results
        return dump_results


if __name__ == '__main__':
    dets = [{'image_id': 0, 'category_id': 1, 'score': 0.9997885823249817,
             'bbox': [101.29035949707031, 58.49554443359375, 285.88792419433594, 666.3601684570312]},
            {'image_id': 0, 'category_id': 1, 'score': 0.9983697533607483,
             'bbox': [487.6786804199219, 102.89520263671875, 113.91610717773438, 440.55474853515625]},
            {'image_id': 0, 'category_id': 1, 'score': 0.9872122406959534,
             'bbox': [357.73284912109375, 193.93605041503906, 129.2513427734375, 257.06639099121094]},
            {'image_id': 0, 'category_id': 1, 'score': 0.9827841520309448,
             'bbox': [276.8230285644531, 79.83059692382812, 112.18902587890625, 485.0297546386719]},
            {'image_id': 0, 'category_id': 1, 'score': 0.08128783106803894,
             'bbox': [126.05856323242188, 68.1190185546875, 468.7314758300781, 468.8319091796875]},
            {'image_id': 0, 'category_id': 1, 'score': 0.06570468842983246,
             'bbox': [279.3220520019531, 91.81890869140625, 86.033447265625, 294.807373046875]},
            {'image_id': 0, 'category_id': 1, 'score': 0.05980807915329933,
             'bbox': [349.9126281738281, 225.49508666992188, 63.00213623046875, 233.75592041015625]},
            {'image_id': 0, 'category_id': 1, 'score': 0.052285727113485336,
             'bbox': [224.21438598632812, 172.93502807617188, 152.7777099609375, 530.9790344238281]},
            {'image_id': 0, 'category_id': 1, 'score': 0.04197293147444725,
             'bbox': [304.7789001464844, 303.732666015625, 85.49383544921875, 253.549072265625]},
            {'image_id': 0, 'category_id': 1, 'score': 0.040261685848236084,
             'bbox': [244.19317626953125, 202.0154266357422, 361.5306396484375, 256.77796936035156]},
            {'image_id': 0, 'category_id': 1, 'score': 0.03856382146477699,
             'bbox': [413.693115234375, 196.3871307373047, 27.736572265625, 27.111602783203125]},
            {'image_id': 0, 'category_id': 1, 'score': 0.034533873200416565,
             'bbox': [275.670654296875, 204.32069396972656, 71.282958984375, 366.0865936279297]},
            {'image_id': 0, 'category_id': 1, 'score': 0.03142502158880234,
             'bbox': [272.6557312011719, 91.92974853515625, 181.243896484375, 336.27850341796875]},
            {'image_id': 0, 'category_id': 1, 'score': 0.029084036126732826,
             'bbox': [364.4385070800781, 113.19952392578125, 222.19235229492188, 439.9608154296875]},
            {'image_id': 0, 'category_id': 1, 'score': 0.027037683874368668,
             'bbox': [358.4462890625, 310.48590087890625, 41.43206787109375, 137.6431884765625]},
            {'image_id': 0, 'category_id': 1, 'score': 0.022109970450401306,
             'bbox': [275.2358093261719, 186.3593292236328, 715.2586975097656, 366.8308563232422]},
            {'image_id': 0, 'category_id': 1, 'score': 0.018988441675901413,
             'bbox': [389.58184814453125, 110.90171813964844, 386.4993896484375, 442.5724639892578]},
            {'image_id': 0, 'category_id': 1, 'score': 0.016994532197713852,
             'bbox': [384.6436767578125, 199.08766174316406, 105.4635009765625, 126.79237365722656]},
            {'image_id': 0, 'category_id': 1, 'score': 0.016872618347406387,
             'bbox': [382.93035888671875, 238.13233947753906, 370.44873046875, 219.0616912841797]},
            {'image_id': 0, 'category_id': 1, 'score': 0.016028564423322678,
             'bbox': [105.85716247558594, 74.07319641113281, 334.02296447753906, 289.17662048339844]},
            {'image_id': 0, 'category_id': 1, 'score': 0.015759054571390152,
             'bbox': [37.17120361328125, 192.1077117919922, 841.8739013671875, 490.1596221923828]}]

    for i in dets:
        i['data'] = cv2.imread ( 'img_%06d.png' % i['image_id'] )

    detector = Detector2D ()
    print ( detector.detect ( dets ) )
