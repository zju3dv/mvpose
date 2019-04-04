
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os


class ModelConfig ( object ):
    model_dir = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ) ) )
    root_dir = os.path.abspath ( os.path.join ( model_dir, '..', '..' ) )
    datasets_dir = os.path.join ( root_dir, 'datasets' )
    shelf_path = os.path.join ( datasets_dir, 'Shelf' )
    campus_path = os.path.join ( datasets_dir, 'CampusSeq1' )
    ultimatum1_path = os.path.join ( datasets_dir, '160422_ultimatum1', 'vgaImgs' )



    shelf_range = range ( 300, 600 )
    campus_range = [i for i in range ( 350, 471 )] + [i for i in range ( 650, 751 )]
    vga_frame_rate = 25
    ultimatum1_range = list ( range ( 17337, 17370 ) ) + list ( range ( 21560, 21660 ) )

    joint_num = 17
    rerank = False
    use_mincut = False
    metric = 'geometry mean'
    testing_on = 'Shelf'
    reprojection_refine = False
    refine_threshold = 1
    semantic_matching = False
    match_SVT = True
    dual_stochastic_SVT = False
    lambda_SVT = 50
    alpha_SVT = 0.5
    eta = 1.5
    beta = 0.5
    use_bundle = False
    spectral = True
    hybrid = True

    def __repr__(self):
        if self.semantic_matching:
            return f'testing_on: {self.testing_on}  eta:{self.eta} metric: {self.metric}'
        elif self.match_SVT:
            return f'testing_on: {self.testing_on}  alpha:{self.alpha_SVT} lambda:{self.lambda_SVT}'
        else:
            return f'testing_on: {self.testing_on}  beta:{self.beta} metric: {self.metric}'


model_cfg = ModelConfig ()
if model_cfg.root_dir not in sys.path:
    sys.path.append ( model_cfg.root_dir )
