import argparse

import bdd_format
from eval import get_official_eval_result, get_coco_eval_result


def parse_args():
    parser = argparse.ArgumentParser(description='Monocular 3D Estimation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('set', choices=['gta', 'kitti'], default='gta')
    parser.add_argument('phase', choices=['train', 'val', 'test'], 
                        default='val')
    parser.add_argument('--root', default='../../',
                        help='root path of the repository')
    parser.add_argument('--data_split', choices=['train', 'val', 'test'], 
                        default='val',
                        help='Which data split to use in testing')
    parser.add_argument('--session', default='616',
                        help='Name of the session, to separate exp')
    parser.add_argument('--epoch', default='030',
                        help='How many epochs you used to separate exp')
    parser.add_argument('--setting', default='age20_aff0.1_hit0_100m_803',
                        help='Name of the result for tracking')
    args = parser.parse_args()

    return args

def show_result(gt_annos, pd_annos, obj_type=[0]):
    # obj_type [0] == Car

    result_kitti = get_official_eval_result(gt_annos, pd_annos, obj_type)
    print(result_kitti)

    result_coco = get_coco_eval_result(gt_annos, pd_annos, obj_type)
    print(result_coco)


def main():
    args = parse_args()

    METHODS = ['none', 'kf3docc', 'lstm']

    if args.set == 'gta':
        gt_name = '{0}data/gta5_tracking/{1}/label/'
        json_name = '*/*_final.json'
    else:
        gt_name = '{0}data/kitti_tracking/{1}ing/label_02/'
        json_name = '*/*.json'
    gt_name = gt_name.format(args.root, args.phase)

    # Load GT annotations
    gt_annos = bdd_format.load_annos_bdd(gt_name, folder=True, json_name=json_name)

    for method in METHODS:
        
        dt_name = '{0}output/{1}_{2}_{3}_{4}_set/{5}_{6}/data/'
        pd_name = dt_name.format(args.root, 
                                args.session, 
                                args.epoch,
                                args.set, 
                                args.phase, 
                                method, 
                                args.setting)
        # Load PD annotations
        pd_annos = bdd_format.load_preds_bdd(pd_name, folder=True)

        # Show KITTI, COCO evaluation results
        show_result(gt_annos, pd_annos)


if __name__ == '__main__':
    main()
