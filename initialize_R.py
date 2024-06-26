import argparse
import glob
import os
import json
import numpy as np
from utils.func import R_from_2poses, matrix_from_quaternion, quaternion_from_matrix

parser = argparse.ArgumentParser(description='PyTorch Train: DetNet')
# Dataset setting
parser.add_argument('--log_dir', type=str, default='pretrain/evaluation_testset/ah',
                    help='save dir of the test logs')
parser.add_argument('--json_path', type=str, default='R.json',
                    help='path to save the estimated rotation matrix')
parser.add_argument('--setup', type=int, default=0, help='id of headset')
parser.add_argument('--pair', type=str, default='0,1', help='id of dual-camera pair')
parser.add_argument('-eid', '--evaluate_id', default=34, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
args = parser.parse_args()
JNUM = 21


def R_from_2hands(hands1, hands2, valids1, valids2, R12=None, R21=None):
    for i in range(hands1.shape[0]):
        pred1, pred2 = hands1[i], hands2[i]
        p1, p2 = [], []
        for j in range(pred1.shape[0]):
            if valids1[i, j] and valids2[i, j]:
                p1.append(pred1[j])
                p2.append(pred2[j])
        if R12 is None:
            R12 = R_from_2poses(p1, p2, is_torch=False)
        if R21 is None:
            R21 = R_from_2poses(p2, p1, is_torch=False)

    return R12, R21


def calc_R(log_path):
    lines = open(log_path).readlines()
    lines.pop(0)
    line_num = len(lines)

    R_ls, R_gt = [], None

    for i in range(min(512, line_num // 2)):
        view0 = lines[2 * i].split()
        view1 = lines[2 * i + 1].split()

        pred0 = np.array(view0[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid0 = np.array(view0[6].split(',')).astype(np.float64).reshape(JNUM, )

        pred1 = np.array(view1[4].split(',')).astype(np.float64).reshape(JNUM, 3)
        valid1 = np.array(view1[6].split(',')).astype(np.float64).reshape(JNUM, )

        R_pred, _ = R_from_2hands(pred0[np.newaxis], pred1[np.newaxis],
                                  valid0[np.newaxis, :], valid1[np.newaxis, :])
        R_ls.append(R_pred)

    quan_ls = [quaternion_from_matrix(R) for R in R_ls]
    quan_avg = np.mean(quan_ls, axis=0)
    quan_avg /= np.linalg.norm(quan_avg)

    R_est = matrix_from_quaternion(np.mean(quan_ls, axis=0))
    return R_est


if __name__ == '__main__':
    eid = args.evaluate_id
    logs = os.path.join(args.log_dir, f'{eid}-set{args.setup}-{args.pair}.log')
    print(logs)

    if os.path.exists(args.json_path):
        with open(args.json_path) as f:
            R_init = json.load(f)
    else:
        R_init = {}

    R_pred = calc_R(logs)

    pair_dict = R_init.get(f"set{args.setup}-{args.pair}", {})
    pair_dict["R_pred"] = R_pred.tolist()
    R_init[f"set{args.setup}-{args.pair}"] = pair_dict
    with open(args.json_path, 'w') as f:
        json.dump(R_init, f, indent=4)
    print(f"Write set{args.setup}-{args.pair} to {args.json_path}.")
