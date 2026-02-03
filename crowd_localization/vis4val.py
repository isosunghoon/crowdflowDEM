# Code with ground truth
# import os
# import numpy as np
# from scipy import spatial as ss
# import cv2
# from misc.utils import hungarian, read_pred_and_gt, AverageMeter, AverageCategoryMeter

# DATASET = 'NWPU'
# DATA_ROOT = '../ProcessedData/' + DATASET
# EXP_NAME = './saved_exp_results/XXX_vis_results'
# PRED_FILE = './saved_exp_results/NWPU_HR_Net_val.txt'
# SAMPLE_ID = 3110

# def main():
#     gt_file = os.path.join(DATA_ROOT, 'val_gt_loc.txt')
#     img_path = os.path.join(DATA_ROOT, 'images')

#     if not os.path.exists(EXP_NAME):
#         os.mkdir(EXP_NAME)

#     pred_data, gt_data = read_pred_and_gt(PRED_FILE, gt_file)

#     sample_id = SAMPLE_ID
#     print(sample_id)
#     gt_p, pred_p, fn_gt_index, tp_pred_index, fp_pred_index, ap, ar = [], [], [], [], [], [], []

#     if gt_data[sample_id]['num'] == 0 and pred_data[sample_id]['num'] != 0:
#         pred_p = pred_data[sample_id]['points']
#         fp_pred_index = np.array(range(pred_p.shape[0]))
#         ap = 0
#         ar = 0

#     if pred_data[sample_id]['num'] == 0 and gt_data[sample_id]['num'] != 0:
#         gt_p = gt_data[sample_id]['points']
#         fn_gt_index = np.array(range(gt_p.shape[0]))
#         sigma_l = gt_data[sample_id]['sigma'][:, 1]
#         ap = 0
#         ar = 0

#     if gt_data[sample_id]['num'] != 0 and pred_data[sample_id]['num'] != 0:
#         pred_p = pred_data[sample_id]['points']
#         gt_p = gt_data[sample_id]['points']
#         sigma_l = gt_data[sample_id]['sigma'][:, 1]
#         level = gt_data[sample_id]['level']

#         dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
#         match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
#         for i_pred_p in range(pred_p.shape[0]):
#             pred_dist = dist_matrix[i_pred_p, :]
#             match_matrix[i_pred_p, :] = pred_dist <= sigma_l

#         tp, assign = hungarian(match_matrix)
#         fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
#         tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
#         tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
#         fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

#         pre = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fp_pred_index.shape[0] + 1e-20)
#         rec = tp_pred_index.shape[0] / (tp_pred_index.shape[0] + fn_gt_index.shape[0] + 1e-20)

#     img = cv2.imread(os.path.join(img_path, str(sample_id) + '.jpg'))

#     point_r_value = 5
#     thickness = 3
#     if gt_data[sample_id]['num'] != 0:
#         for i in range(gt_p.shape[0]):
#             if i in fn_gt_index:
#                 cv2.circle(img, (gt_p[i][0], gt_p[i][1]), point_r_value, (0, 0, 255), -1)
#                 cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 0, 255), thickness)
#             else:
#                 cv2.circle(img, (gt_p[i][0], gt_p[i][1]), sigma_l[i], (0, 255, 0), thickness)
#     if pred_data[sample_id]['num'] != 0:
#         for i in range(pred_p.shape[0]):
#             if i in tp_pred_index:
#                 cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value, (0, 255, 0), -1)
#             else:
#                 cv2.circle(img, (pred_p[i][0], pred_p[i][1]), point_r_value * 2, (255, 0, 255), -1)

#     cv2.imwrite(os.path.join(EXP_NAME, str(sample_id) + '_pre_' + str(pre)[0:6] + '_rec_' + str(rec)[0:6] + '.jpg'), img)

# if __name__ == '__main__':
#     main()

# Code with no ground truth
import os
import numpy as np
import cv2
from misc.utils import read_pred_and_gt

# Constants for configuration
DATASET = 'NWPU'
DATA_ROOT = '../ProcessedData/' + DATASET
EXP_NAME = './saved_exp_results/XXX_vis_results'
TEST_LIST = 'new.txt'  # File containing list of image filenames to process
PRED_FILE = './saved_exp_results/NWPU_HR_Net_'+TEST_LIST

def main():
    img_path = os.path.join(DATA_ROOT, 'images')

    if not os.path.exists(EXP_NAME):
        os.mkdir(EXP_NAME)

    # Read prediction data
    pred_data, _ = read_pred_and_gt(PRED_FILE)

    # Read the list of image filenames to process
    with open(os.path.join(DATA_ROOT, TEST_LIST), 'r') as f:
        img_filenames = [line.strip() for line in f]

    for filename in img_filenames:
        sample_id = int(filename.split('.')[0])
        print(sample_id)
        pred_p = []

        if pred_data[sample_id]['num'] != 0:
            pred_p = pred_data[sample_id]['points']

        # Load the image
        img = cv2.imread(os.path.join(img_path, filename+'.jpg'))

        point_r_value = 5

        # Draw circles for predicted points
        if pred_data[sample_id]['num'] != 0:
            for i in range(pred_p.shape[0]):
                cv2.circle(img, (int(pred_p[i][0]), int(pred_p[i][1])), point_r_value, (255, 0, 0), -1)  # Blue for predicted points

        # Save the visualization result
        cv2.imwrite(os.path.join(EXP_NAME, f'{sample_id}_pred.jpg'), img)

if __name__ == '__main__':
    main()
