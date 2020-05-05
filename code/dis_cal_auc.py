from scipy.io import loadmat
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

'''
test_mat_file='../../test_data/test2_annotations.mat'
prediction_file = '../npzs/multi_scale_concat_heatmaps.npz'

anns = loadmat(test_mat_file)
gazes = anns['test_gaze']
eyes = anns['test_eyes']
N = anns['test_path'].shape[0]

prediction = np.load(prediction_file)['heatmaps']
print(prediction.shape)

gt_list, pred_list = [], []
error_list = []
for i in range(N):
    pred = prediction[i, :, :]
    eye_point = eyes[0, i][0]
    gt_points = gazes[0, i]
    pred = cv2.resize(pred, (5, 5))
    #pred[...] = 0.0
    #pred[2, 2] = 1.0
    gt_heatmap = np.zeros((5, 5))
    for gt_point in gt_points:
        x, y = list(map(int, list(gt_point * 5)))
        gt_heatmap[y, x] = 1.0

    score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
    error_list.append(score)
    gt_list.append(gt_heatmap)
    pred_list.append(pred)

print("mean", np.mean(error_list))
gt_list = np.stack(gt_list).reshape([-1])
pred_list = np.stack(pred_list).reshape([-1])

print("auc score")
score = roc_auc_score(gt_list, pred_list)
print(score)
'''

test_mat_file='../../test_data/test2_annotations.mat'
prediction_file = '../npzs/multi_scale_concat_heatmaps.npz'

mat = loadmat(test_mat_file)
N = mat['test_path'].shape[0]

gazes_list = [ [] for _ in range(16)]
eyes_list = [ [] for _ in range(16) ]

for i in range(N):
    eye_x, eye_y = mat['test_eyes'][0][i][0][0], mat['test_eyes'][0][i][0][1]
    r_idx, c_idx = int(eye_y / 0.25), int(eye_x / 0.25)
    w_idx = r_idx * 4 + c_idx

    gazes_list[w_idx].append(mat['test_gaze'][0][i])
    eyes_list[w_idx].append(mat['test_eyes'][0][i])

'''
anns = loadmat(test_mat_file)
gazes = anns['test_gaze']
eyes = anns['test_eyes']
N = anns['test_path'].shape[0]

prediction = np.load(prediction_file)['heatmaps']
print(prediction.shape)

gt_list, pred_list = [], []
error_list = []
for i in range(N):
    pred = prediction[i, :, :]
    eye_point = eyes[0, i][0]
    gt_points = gazes[0, i]
    pred = cv2.resize(pred, (5, 5))
    #pred[...] = 0.0
    #pred[2, 2] = 1.0
    gt_heatmap = np.zeros((5, 5))
    for gt_point in gt_points:
        x, y = list(map(int, list(gt_point * 5)))
        gt_heatmap[y, x] = 1.0

    score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
    error_list.append(score)
    gt_list.append(gt_heatmap)
    pred_list.append(pred)

print("mean", np.mean(error_list))
gt_list = np.stack(gt_list).reshape([-1])
pred_list = np.stack(pred_list).reshape([-1])

score = roc_auc_score(gt_list, pred_list)
print("auc score", score)
'''

means_list, aucs_list = [], []
for dataset_idx in range(16):
    prediction_file = '../npzs/multi_scale_concat_heatmaps_{}.npz'.format(str(dataset_idx))
    
    prediction = np.load(prediction_file)['heatmaps']
    #print(prediction.shape)
    
    gt_list, pred_list = [], []
    error_list = []
    for i in range(prediction.shape[0]):
        pred = prediction[i, :, :]
        #eye_point = eyes[0, i][0]
        #gt_points = gazes[0, i]
        eye_point = eyes_list[dataset_idx][i][0]
        gt_points = gazes_list[dataset_idx][i]
        pred = cv2.resize(pred, (5, 5))
        #pred[...] = 0.0
        #pred[2, 2] = 1.0
        gt_heatmap = np.zeros((5, 5))
        for gt_point in gt_points:
            x, y = list(map(int, list(gt_point * 5)))
            gt_heatmap[y, x] = 1.0
    
        score = roc_auc_score(gt_heatmap.reshape([-1]).astype(np.int32), pred.reshape([-1]))
        error_list.append(score)
        gt_list.append(gt_heatmap)
        pred_list.append(pred)
    
    print("mean", np.mean(error_list))
    gt_list = np.stack(gt_list).reshape([-1])
    pred_list = np.stack(pred_list).reshape([-1])
    
    score = roc_auc_score(gt_list, pred_list)
    print("auc score", score)

    means_list.append(np.mean(error_list))
    aucs_list.append(score)

print("Mean value:", np.mean(np.array(means_list)))
print("AUC score:", np.mean(np.array(aucs_list)))
    
    
