import numpy as np
from tools.label_constants import *

UNKNOWN_ID = 255
NO_FEATURE_ID = 255

def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]
    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    '''calculate IoU.'''
    tp = np.longlong(confusion[label_id, label_id])
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom, fn + tp

def evaluate(pred_ids, gt_ids, stdout=True, dataset='replica', return_class = False):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'replica' in dataset:
        CLASS_LABELS = Replica_Evaluate_label30
    elif 'mp_label35' in dataset:
        CLASS_LABELS = Matterport_Evaluate_label35
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (class_ious[label_name][3]).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    N = N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))
                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d}) ; {4:>5.3f} ({5:>6d}/{6:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2],
                        class_accs[label_name],
                        class_ious[label_name][1],
                        class_ious[label_name][3],))
            except:
                print(label_name + ' error!')
                N = N - 1
                continue

        mean_iou /= N
        mean_acc /= N
        print('Mean IoU:', mean_iou)
        print('Mean Acc:', mean_acc)
    
    if return_class:
        return mean_iou, mean_acc, class_ious, class_accs
    else:
        return mean_iou, mean_acc
