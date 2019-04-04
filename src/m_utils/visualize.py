
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import copy
import random
import numpy as np
import cv2
import os.path as osp


def drawlines(img0, img1, lines, pts0, pts1):
    """

    :param img0: img where the projected points are
    :param img1: img to plot lines
    :param lines: lines from opencv
    :param pts0: numpy array of points
    :param pts1: numpy array of points
    :return:
    """
    r, c, _ = img1.shape
    for r, pt0, pt1 in zip ( lines, pts0, pts1 ):
        color = tuple ( np.random.randint ( 0, 255, 3 ).tolist () )
        x0, y0 = map ( int, [0, -r[2] / r[1]] )
        x1, y1 = map ( int, [c, -(r[2] + r[0] * c) / r[1]] )
        img0 = cv2.circle ( img0, tuple ( pt0 ), 10, color, -1 )
        img1 = cv2.circle ( img1, tuple ( pt1 ), 10, color, -1 )
        img1 = cv2.line ( img1, (x0, y0), (x1, y1), color, 2 )
    return img0, img1


def adjust_imgae_plot(img, title=None):
    """
    util to hide label
    :param img:
    :param title:
    :return:
    """
    plt.imshow ( img )
    plt.xticks ( [] )
    plt.yticks ( [] )
    plt.title ( title )


def show_epilines(img_cv0, img_cv1, F_01, pts_0, pts_1, show=True):
    """

    :param img_cv0:
    :param img_cv1:
    :param F_01:
    :param pts_0:
    :param pts_1:
    :param show:
    :return:
    """
    lines1 = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2, F_01 )
    lines1 = lines1.reshape ( -1, 3 )
    img_draw_0, img_draw_1 = drawlines ( img_cv0.copy (), img_cv1.copy (), lines1, pts_0, pts_1 )
    img_draw_0, img_draw_1 = cv2.cvtColor ( img_draw_0, cv2.COLOR_BGR2RGB ), cv2.cvtColor ( img_draw_1,
                                                                                            cv2.COLOR_BGR2RGB )
    return img_draw_0, img_draw_1


def visualize(img, det_box_list=None, gt_box_list=None, keypoints_list=None,
              show_skeleton_labels=False, return_img=False):
    im = np.array ( img ).copy ().astype ( np.uint8 )
    im = cv2.cvtColor ( im, cv2.COLOR_RGB2BGR )  # Note: assume image read from PIL.Image
    if det_box_list:
        for det_boxes in det_box_list:
            det_boxes = np.array ( det_boxes )
            bb = det_boxes[:4].astype ( int )
            cv2.rectangle ( im, (bb[0], bb[1]), (bb[0] + bb[2], bb[0] + bb[3]),
                            (0, 0, 255),
                            5 )

    if gt_box_list:
        for gt_boxes in gt_box_list:
            gt_boxes = np.array ( gt_boxes )
            for gt in gt_boxes:
                bb = gt[:4].astype ( int )
                cv2.rectangle ( im, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 3 )
    if keypoints_list:
        for keypoints in keypoints_list:
            keypoints = np.array ( keypoints ).astype ( int )
            keypoints = keypoints.reshape ( -1, 17, 3 )

            for i in range ( len ( keypoints ) ):
                draw_skeleton ( im, keypoints[i], show_skeleton_labels )

    im = cv2.cvtColor ( im, cv2.COLOR_BGR2RGB )

    if return_img:
        return im.copy ()
    else:
        plt.imshow ( im )
        plt.show ()


def draw_skeleton(aa, kp, show_skeleton_labels=False):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    for i, j in skeleton:
        if kp[i - 1][0] >= 0 and kp[i - 1][1] >= 0 and kp[j - 1][0] >= 0 and kp[j - 1][1] >= 0 and \
                (len ( kp[i - 1] ) <= 2 or (len ( kp[i - 1] ) > 2 and kp[i - 1][2] > 0.1 and kp[j - 1][2] > 0.1)):
            cv2.line ( aa, tuple ( kp[i - 1][:2] ), tuple ( kp[j - 1][:2] ), (0, 255, 255), 5 )
    for j in range ( len ( kp ) ):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len ( kp[j] ) <= 2 or (len ( kp[j] ) > 2 and kp[j][2] > 1.1):
                cv2.circle ( aa, tuple ( kp[j][:2] ), 5, tuple ( (0, 0, 255) ), -1 )
            elif len ( kp[j] ) <= 2 or (len ( kp[j] ) > 2 and kp[j][2] > 0.1):
                cv2.circle ( aa, tuple ( kp[j][:2] ), 5, tuple ( (255, 0, 0) ), -1 )

            if show_skeleton_labels and (len ( kp[j] ) <= 2 or (len ( kp[j] ) > 2 and kp[j][2] > 0.1)):
                cv2.putText ( aa, kp_names[j], tuple ( kp[j][:2] ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0) )


def plot_pose3d(pose):
    """Plot the 3D pose showing the joint connections.
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    """
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                   [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [5, 7, 9]:
            _c = 1
        if j in [6, 8, 10]:
            _c = 2
        if j in [11, 13, 15]:
            _c = 3
        if j in [12, 14, 16]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure ()
    ax = fig.gca ( projection='3d' )
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color ( c[0] )
        ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                  [pose[1, c[0]], pose[1, c[1]]],
                  [pose[2, c[0]], pose[2, c[1]]], c=col )
    for j in range ( pose.shape[1] ):
        col = '#%02x%02x%02x' % joint_color ( j )
        ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                     c=col, marker='o', edgecolor=col )
    smallest = pose.min ()
    largest = pose.max ()
    ax.set_xlim3d ( smallest, largest )
    ax.set_ylim3d ( smallest, largest )
    ax.set_zlim3d ( smallest, largest )

    return fig


def plot_multi_pose3d(poses, inplace=True):
    """Plot the 3D pose showing the joint connections.
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    R = np.array ( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] )
    poses = [R @ i for i in poses]
    _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                   [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [5, 7, 9]:
            _c = 1
        if j in [6, 8, 10]:
            _c = 2
        if j in [11, 13, 15]:
            _c = 3
        if j in [12, 14, 16]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    fig = plt.figure ()
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    if inplace:
        ax = fig.gca ( projection='3d' )
    else:
        ax = fig.add_subplot ( rows, rows, 1, projection='3d' )

    smallest = [min ( [i[idx].min () for i in poses] ) for idx in range ( 3 )]
    largest = [max ( [i[idx].max () for i in poses] ) for idx in range ( 3 )]
    ax.set_xlim3d ( smallest[0], largest[0] )
    ax.set_ylim3d ( smallest[1], largest[1] )
    ax.set_zlim3d ( smallest[2], largest[2] )

    for i, pose in enumerate ( poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='o', edgecolor=col )
        ax.set_label ( f'#{i}' )
    return fig


def show_panel(dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat, geo_affinity_mat, W,
               multi_pose3d, chosen_img):
    for i, cam_id in enumerate ( dataset.cam_names ):
        info_dict = copy.deepcopy ( dataset.info_dict[cam_id][str ( img_id )] )
        fname = info_dict.pop ( 'image_name', None )
        img = Image.open ( osp.join ( dataset.data_dir, fname ) )
        img = visualize ( img,
                          keypoints_list=[i['pose2d'] for _, i in info_dict.items ()],
                          return_img=True )
        plt.subplot ( 6, 6, 6 * i + 1 )
        plt.imshow ( img )
        plt.xlabel ( f'{osp.split(osp.split(fname)[-2])[-1]}/{img_id}' )
        plt.xticks ( [] )
        plt.yticks ( [] )

    for i, person in enumerate ( matched_list ):
        # Plot origin image
        for sub_imageid in person:
            cam_id = sub_imgid2cam[sub_imageid]
            img = Image.open ( osp.join ( dataset.data_dir, info_list[sub_imageid]['img_path'] ) )
            plt.subplot ( 6, 6, cam_id * 6 + i + 2 )
            plt.imshow ( img )
            plt.xlabel ( f"#{sub_imageid}#{'Checked' if sub_imageid in chosen_img[i] else 'No'}" )
            plt.xticks ( [] )
            plt.yticks ( [] )

    ax = plt.subplot ( 6, 6, 31 )
    sns.heatmap ( affinity_mat )
    ax.set_title ( 'ReID' )
    ax = plt.subplot ( 6, 6, 32 )
    sns.heatmap ( geo_affinity_mat )
    ax.set_title ( 'Geometry' )
    ax = plt.subplot ( 6, 6, 33 )
    sns.heatmap ( W )
    ax.set_title ( 'Synthesize' )

    fig = plot_multi_pose3d ( multi_pose3d )
    fig.suptitle ( f'Image: {img_id}' )
    fig.show ()
    plt.tight_layout ()
    plt.show ()


def plot_shelf_multi_pose3d(poses, inplace=False):
    """
    Plot the 3D pose showing the joint connections.
    shelf_joint_name = ['Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', #6
                        'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Bottom Head', #12
                        'Top Head']
    """
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[0, 1], [1, 2], [2, 3], [2, 8], [3, 9], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
                   [12, 13]]
    maker = ['o', 'x']

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [3, 4, 5]:
            _c = 1
        if j in [6, 7, 8]:
            _c = 2
        if j in [9, 10, 11]:
            _c = 3
        if j in [12, 13]:
            _c = 4

        return colors[_c]

    fig = plt.figure ()
    if inplace:
        ax = fig.gca ( projection='3d' )
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    for i, pose in enumerate ( poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker=maker[i % 2], edgecolor=col )
        smallest = pose.min ()
        largest = pose.max ()
        ax.set_xlim3d ( smallest, largest )
        ax.set_ylim3d ( smallest, largest )
        ax.set_zlim3d ( smallest, largest )
        ax.set_label ( f'#{i}' )
    return fig


def show_panel_mem(dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat, geo_affinity_mat, W,
                   plt_id, multi_pose3d):
    try:
        show_panel_mem.counter += 1
    except AttributeError:
        show_panel_mem.counter = 0
    cols = len ( matched_list ) + 1
    rows = sub_imgid2cam.max () + 2
    Ps = dataset.P
    reprojectedPoses = list ()
    for camId, P in enumerate ( Ps ):
        reprojectedPoses.append ( [] )
        for pose3d in multi_pose3d:
            pose3dHomo = np.ones ( (4, pose3d.shape[1]) )
            pose3dHomo[:3] = pose3d
            pose2dHomo = P @ pose3dHomo
            pose2dHomo /= pose2dHomo[2]
            reprojectedPoses[camId].append ( pose2dHomo.T )

    for i, cam_id in enumerate ( dataset.cam_names ):
        info_dict = copy.deepcopy ( dataset.info_dict[cam_id][img_id] )
        img = dataset.info_dict[cam_id]['image_data']
        imgDeteced = visualize ( img,
                                 keypoints_list=[i['pose2d'] for i in info_dict],
                                 det_box_list=[i['bbox'] for i in info_dict],
                                 return_img=True )
        imgProjected = visualize ( img,
                                   keypoints_list=reprojectedPoses[i],
                                   return_img=True )
        plt.subplot ( rows, cols, cols * i + 1 )
        plt.imshow ( imgProjected )
        plt.xlabel ( f'{cam_id}/{plt_id}' )
        # img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite ( f'./result/dump_img/{cam_id}#{plt_id}.png', img_ )
        plt.xticks ( [] )
        plt.yticks ( [] )

    for i, person in enumerate ( matched_list ):
        # Plot origin image
        for sub_imageid in person:
            cam_id = sub_imgid2cam[sub_imageid]
            img = info_list[sub_imageid]['cropped_img']
            plt.subplot ( rows, cols, cam_id * cols + i + 2 )
            plt.imshow ( img )
            # img_ = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite ( f"./result/dump_img/#{sub_imageid}.png", img_ )
            plt.xlabel ( f"#{sub_imageid}" )
            plt.xticks ( [] )
            plt.yticks ( [] )
    ax = plt.subplot ( rows, cols, (rows - 1) * cols + 1 )
    sns.heatmap ( affinity_mat )
    ax.set_title ( 'ReID' )
    ax = plt.subplot ( rows, cols, (rows - 1) * cols + 2 )
    sns.heatmap ( geo_affinity_mat )
    ax.set_title ( 'Geometry' )
    ax = plt.subplot ( rows, cols, (rows - 1) * cols + 3 )
    sns.heatmap ( W )
    ax.set_title ( 'Synthesize' )

    if False and multi_pose3d:
        fig = plot_multi_pose3d ( multi_pose3d )
        fig.suptitle ( f'Image: {img_id}' )
        fig.show ()
    # plt.savefig ( f'./result/Shelf_panel/{plt_id}.png' )
    # plt.close()
    plt.show ()


def plot_panoptic(poses, x_poses=list (), inplace=True):
    """Plot the 3D pose showing the joint connections.
    panoptic_joints = ['Neck', 'HeadTop', 'BodyCenter', 'lShoulder','lElbow', 'lWrist', #5
                   'lHip', 'lKnee', 'lAnkle', 'rShoulder', 'rElbow',#10
                   'rWrist', 'rHip', 'rKnee', 'rAnkle']
    """
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[0, 1], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11], [0, 2], [2, 12], [12, 13], [13, 14],
                   [2, 6], [6, 7], [7, 8]]

    def joint_color(j):
        # TODO: change joint color
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [3, 4, 5]:
            _c = 1
        if j in [6, 7, 8]:
            _c = 2
        if j in [9, 10, 11]:
            _c = 3
        if j in [12, 13, 14]:
            _c = 4
        # if j in range ( 14, 17 ):
        #     _c = 5
        return colors[_c]

    fig = plt.figure ()
    if inplace:
        ax = fig.gca ( projection='3d' )
        # ax.view_init ( 90, 90 )  # To adjust wired world coordinate
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    for i, pose in enumerate ( poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='o', edgecolor=col )
        smallest = pose.min ()
        largest = pose.max ()
        ax.set_xlim3d ( smallest, largest )
        ax.set_ylim3d ( smallest, largest )
        ax.set_zlim3d ( smallest, largest )
        ax.set_label ( f'#{i}' )
    for i, pose in enumerate ( x_poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle=':' )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='x', edgecolor=col )
        smallest = pose.min ()
        largest = pose.max ()
        ax.set_xlim3d ( smallest, largest )
        ax.set_ylim3d ( smallest, largest )
        ax.set_zlim3d ( smallest, largest )
        ax.set_label ( f'#{i}' )
    return fig


def plot_coco19(poses, x_poses=list (), inplace=True):
    """Plot the 3D pose showing the joint connections.
    coco19_kp_names = ['neck', 'nose', 'hip', 'l_shoulder', 'l_elbow', 'l_wrist',  # 5
                'l_hip', 'l_knee', 'l_ankle', 'r_shoulder', 'r_elbow',  # 10
                'r_wrist', 'r_hip', 'r_knee', 'r_ankle', 'l_eye', # 15
                'l_ear', 'r_eye', 'r_ear']
    """
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [[0, 1], [1, 15], [1, 17], [15, 16], [17, 18], [0, 2], [2, 6], [6, 7], [7, 8], [2, 12], [12, 13],
                   [13, 14],
                   [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11]]

    def joint_color(j):
        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in [3, 4, 5]:
            _c = 1
        if j in [6, 7, 8]:
            _c = 2
        if j in [9, 10, 11]:
            _c = 3
        if j in [12, 13, 14]:
            _c = 4
        if j in [15, 17]:
            _c = 5
        return colors[_c]

    fig = plt.figure ()
    if inplace:
        ax = fig.gca ( projection='3d' )
        # ax.view_init ( 90, 90 )  # To adjust wired world coordinate
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    for i, pose in enumerate ( poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='o', edgecolor=col )
        smallest = pose.min ()
        largest = pose.max ()
        ax.set_xlim3d ( smallest, largest )
        ax.set_ylim3d ( smallest, largest )
        ax.set_zlim3d ( smallest, largest )
        ax.set_label ( f'#{i}' )
    for i, pose in enumerate ( x_poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        if not inplace:
            ax = fig.add_subplot ( rows, rows, i + 1, projection='3d' )
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color ( c[0] )
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col, linestyle=':' )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % joint_color ( j )
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='x', edgecolor=col )
        smallest = pose.min ()
        largest = pose.max ()
        ax.set_xlim3d ( smallest, largest )
        ax.set_ylim3d ( smallest, largest )
        ax.set_zlim3d ( smallest, largest )
        ax.set_label ( f'#{i}' )
    return fig


def plotPaperRows(dataset, matched_list, info_list, sub_imgid2cam, img_id, affinity_mat, geo_affinity_mat, W,
                  plt_id, multi_pose3d, saveImg=False):
    try:
        show_panel_mem.counter += 1
    except AttributeError:
        show_panel_mem.counter = 0
    cols = len ( dataset.cam_names )
    rows = 3
    all_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 215, 0), (0, 255, 255), (255, 255, 0)]
    Ps = dataset.P
    reprojectedPoses = list ()
    for camId, P in enumerate ( Ps ):
        reprojectedPoses.append ( [] )
        for pose3d in multi_pose3d:
            pose3dHomo = np.ones ( (4, pose3d.shape[1]) )
            pose3dHomo[:3] = pose3d
            pose2dHomo = P @ pose3dHomo
            pose2dHomo /= pose2dHomo[2]
            reprojectedPoses[camId].append ( pose2dHomo.T )

    def subImgID2Pid(pid):
        for gPid, subImgIds in enumerate ( matched_list ):
            if pid in subImgIds:
                return gPid

    for camIdx, cam_id in enumerate ( dataset.cam_names ):
        info_dict = copy.deepcopy ( dataset.info_dict[cam_id][img_id] )
        poseIDInCam = [subImgID2Pid ( idx ) for idx, camID in enumerate ( sub_imgid2cam ) if cam_id == camID]
        colorAssignment = [all_color[pid_g] for pid_g in poseIDInCam]
        img = dataset.info_dict[cam_id]['image_data']
        imgDetected = visualizeSkeletonPaper ( img, [(255, 255, 255) for _ in info_dict],
                                               keypoints_list=[i['pose2d'] for i in info_dict],
                                               det_box_list=[i['bbox'] for i in info_dict] )
        imgMatched = visualizeSkeletonPaper ( img, colorAssignment, keypoints_list=[i['pose2d'] for i in info_dict],
                                              det_box_list=[i['bbox'] for i in info_dict] )
        imgProjected = visualizeSkeletonPaper ( img, all_color,
                                                keypoints_list=reprojectedPoses[camIdx] )
        plt.subplot ( rows, cols, 1 + camIdx )
        plt.imshow ( imgDetected )
        plt.xticks ( [] )
        plt.yticks ( [] )
        plt.subplot ( rows, cols, 1 + camIdx + cols )
        plt.imshow ( imgMatched )
        plt.xticks ( [] )
        plt.yticks ( [] )
        plt.subplot ( rows, cols, 1 + camIdx + cols * 2 )
        plt.imshow ( imgProjected )
        plt.xticks ( [] )
        plt.yticks ( [] )
        if saveImg:
            cv2.imwrite ( f"cam{camIdx}Detected.png", imgDetected )
            cv2.imwrite ( f"cam{camIdx}Matched.png", imgMatched )
            cv2.imwrite ( f"cam{camIdx}Projected.png", imgProjected )

    if multi_pose3d:
        fig = plotPaper3d ( multi_pose3d, all_color )
        # fig.suptitle ( f'Image: {img_id}' )
        fig.show ()
    # plt.savefig ( f'./result/Shelf_panel/{plt_id}.png' )
    # plt.close()
    plt.show ()


def plotPaper3d(poses, colors):
    """Plot the 3D pose showing the joint connections.
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    # R = np.array ( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] )
    R = np.array ( [[1, 0, 0], [0, 1, 0], [0, 0, 1]] )
    poses = [R @ i for i in poses]
    _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                   [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    fig = plt.figure ()
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    ax = fig.gca ( projection='3d' )

    smallest = [min ( [i[idx].min () for i in poses] ) for idx in range ( 3 )]
    largest = [max ( [i[idx].max () for i in poses] ) for idx in range ( 3 )]
    ax.set_xlim3d ( smallest[0], largest[0] )
    ax.set_ylim3d ( smallest[1], largest[1] )
    ax.set_zlim3d ( smallest[2], largest[2] )

    for i, pose in enumerate ( poses ):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='o', edgecolor=col )
        # ax.set_label ( f'#{i}' )
    return fig


def visualizeSkeletonPaper(img, colors, det_box_list=None, keypoints_list=None, ):
    im = np.array ( img ).copy ().astype ( np.uint8 )
    # im = cv2.cvtColor ( im, cv2.COLOR_RGB2BGR )  # Note: assume image read from PIL.Image
    if det_box_list:
        for boxIdx, det_boxes in enumerate ( det_box_list ):
            det_boxes = np.array ( det_boxes )
            bb = det_boxes[:4].astype ( int )
            cv2.rectangle ( im, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                            colors[boxIdx],
                            5 )

    if keypoints_list:
        for pid, keypoints in enumerate ( keypoints_list ):
            keypoints = np.array ( keypoints ).astype ( int )
            keypoints = keypoints.reshape ( -1, 17, 3 )

            for i in range ( len ( keypoints ) ):
                drawSkeletonPaper ( im, keypoints[i], colors[pid] )

    # im = cv2.cvtColor ( im, cv2.COLOR_BGR2RGB )

    return im.copy ()


def drawSkeletonPaper(img, kp, colors):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    for idx, (i, j) in enumerate ( skeleton ):
        if kp[i - 1][0] >= 0 and kp[i - 1][1] >= 0 and kp[j - 1][0] >= 0 and kp[j - 1][1] >= 0 and \
                (len ( kp[i - 1] ) <= 2 or (len ( kp[i - 1] ) > 2 and kp[i - 1][2] > 0.1 and kp[j - 1][2] > 0.1)):
            cv2.line ( img, tuple ( kp[i - 1][:2] ), tuple ( kp[j - 1][:2] ), colors, 5 )
    for j in range ( len ( kp ) ):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len ( kp[j] ) <= 2 or (len ( kp[j] ) > 2 and kp[j][2] > 1.1):
                cv2.circle ( img, tuple ( kp[j][:2] ), 5, tuple ( colors ), -1 )
            elif len ( kp[j] ) <= 2 or (len ( kp[j] ) > 2 and kp[j][2] > 0.1):
                cv2.circle ( img, tuple ( kp[j][:2] ), 5, tuple ( colors ), -1 )
