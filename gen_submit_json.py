import glob
import json
import os.path
import argparse

import numpy as np
from datetime import datetime

JNUM = 21

cam_setups = [['HMC_21176875', 'HMC_21176623', 'HMC_21110305', 'HMC_21179183'],
              ['HMC_84346135', 'HMC_84347414', 'HMC_84355350', 'HMC_84358933']]


def generate_pred(log_dir, savepath, epoch=10):
    # An example to generate submission json file of a certain epoch
    gt_dict = {"info":
        {
            "description": "AssemblyHands",
            "version": "eccv24-v1.0",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "annotations": {}
    }

    logs = glob.glob(os.path.join(log_dir, f'{epoch}*.log'))
    # print(logs)

    for log in logs:
        print(log)
        basename = os.path.basename(log).split('.')[0]
        setup, pair = basename.split('-')[1:]
        setup = eval(setup[-1])
        pair = list(map(int, pair.split(',')))
        pairname = '+'.join([cam_setups[setup][c] for c in pair])

        lines = open(log).readlines()
        lines.pop(0)
        lines.pop(-1)

        for line in lines:
            line = line.split()
            seq, cam, frame, pred, valid = line[1], line[2], line[3], line[4], line[6]

            pred = np.array(pred.split(",")).astype(np.float64).reshape(JNUM, 3).tolist()

            pair_dic = gt_dict['annotations'].get(pairname, {})
            seq_dic = pair_dic.get(seq, {})
            cam_dic = seq_dic.get(cam, {})
            cam_dic[frame] = pred
            seq_dic[cam] = cam_dic
            pair_dic[seq] = seq_dic
            gt_dict['annotations'][pairname] = pair_dic

    with open(savepath, 'w') as f:
        json.dump(gt_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Train: DetNet')
    # Dataset setting
    parser.add_argument('--evaluation_dir', type=str, default='in_dataset_adapt/evaluation/ah',
                        help='save dir of the test logs')
    parser.add_argument('-eid', '--evaluate_id', default=10, type=int, metavar='N',
                        help='number of epoch to generate submission file')
    parser.add_argument('--savepath', type=str, default='./assemblyhands_test_joint_3d_eccv24_pred.json',
                        help='path to save the submission file')
    args = parser.parse_args()
    generate_pred(args.evaluation_dir, args.savepath, epoch=args.evaluate_id)
