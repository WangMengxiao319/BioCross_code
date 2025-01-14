import numpy as np
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc, accuracy_score, f1_score,matthews_corrcoef,jaccard_score,  hamming_loss, multilabel_confusion_matrix,average_precision_score
import time 
import torch
import pandas as pd
def compute_F1s_threshold(gt, pred, threshold, n_class=6):
    bert_f1 = 0.0
    gt_np = gt.copy()
    pred_np = pred.copy()

    F1s = []
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i], average='binary'))
    F1_macro = np.mean(np.array(F1s))
    return F1s,F1_macro


def compute_Accs_threshold(gt, pred, threshold, n_class=6):
    '''
    这样计算的Acc相当element-acc，而不是subset acc
    '''
    gt_np = gt.copy()
    pred_np = pred.copy()
    # print('logits',pred)
    # print('select_best_thresholds',threshold)
    Accs = []
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    acc_mean = np.mean(np.array(Accs))
    acc_sample = accuracy_score(gt_np, pred_np)
    # print('true',gt_np)
    # print('pred',pred_np)
    # np.savetxt("pred_np.csv",pred_np)
    return Accs,acc_mean,acc_sample


def compute_mccs(gt, pred, n_class=6):
    # get a best threshold for all classes
    gt_np = gt.copy()
    pred_np = pred.copy()
    select_best_thresholds = []
    best_mcc = 0.0

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:, i][pred_np_[:, i] >= thresholds[i]] = 1
            pred_np_[:, i][pred_np_[:, i] < thresholds[i]] = 0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)
    for i in range(n_class):
        '''这里相当于对元素进行了操作，而pred_np又完全指向pred，所以会改变实参'''
        pred_np[:, i][pred_np[:, i] >= select_best_thresholds[i]] = 1
        pred_np[:, i][pred_np[:, i] < select_best_thresholds[i]] = 0
    mccs = []
    # mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs))
    # mccs.append(mean_mcc)
    return mccs, select_best_thresholds
	

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    metrics = {}
    AUROCs = []
    gt_np = gt.copy()
    pred_np = pred.copy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    for idx in range(n_class):
        metrics[f"auc/class_{idx}"] = AUROCs[idx]
    return AUROCs, metrics

def round_compute(prev,threshold,n_class):
    pred = prev.copy()
    # prev1 = np.zeros((len(prev), len(prev[0])))
    # for i in range(len(prev)):
    #     for j in range(len(prev[i])):
    #         if prev[i][j] >= threshold:
    #             prev1[i][j] = 1
    #         else:
    #             prev1[i][j] = 0
    # return prev1
    for i in range(n_class):
        '''这里相当于对元素进行了操作，而pred_np又完全指向pred，所以会改变实参'''
        pred[:, i][pred[:, i] >= threshold[i]] = 1
        pred[:, i][pred[:, i] < threshold[i]] = 0
    return pred
# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values
# Load weights.(cinc2020)
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

def confusion_m(truth_flat, pred_flat):
    num_recordings, n_dim = np.shape(truth_flat)
    if n_dim==1:
        '''binary confusion matrix'''
        num_recordings = truth_flat.shape[0]
        confusion_m1 = np.zeros((2, 2))

        for i in range(num_recordings):
            true_label = truth_flat[i]
            pred_label = pred_flat[i]

            confusion_m1[true_label][pred_label] += 1

    else:
        '''multi class confusion matrix''' # 只考虑了每个样本有一个真实标签的情况
        confusion_m1 = np.zeros((n_dim, n_dim))
        label_num = truth_flat.sum(axis=1)  # 这个函数只考虑了每个样本有一个真实标签的情况
        index = np.where(label_num == 1)[0]

        pred_num = pred_flat[index].sum(axis=1)
        # print(Counter(pred_num))
        for i in index:
            #         for i,pred_f in enumerate(pred_flat[index]):
            true_dis = np.where(truth_flat[i] == 1)[0][0]
            pred_dis = np.where(pred_flat[i] == 1)[0]
            for n in pred_dis:
                confusion_m1[true_dis][n] += 1
        #     print(confusion_m1)
    return confusion_m1

def compute_modified_confusion_matrix(labels, outputs):
    ''' Compute a binary multi-class, multi-label confusion matrix, where the rows
    are the labels and the columns are the outputs.'''
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))  # 相当于求并集之后，再求和，例如对于真实标签[1, 0, 1]和预测输出[1, 0, 0]，正规化因子是2，意味着预测输出中的每一个正标签在混淆矩阵中的权重是1/2。
        # Iterate over all the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score
def calculate_f1(confusion_matrix):
    '''FIXME:f1 what?'''
    num_classes = confusion_matrix.shape[0]
    f1_scores = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)

    return f1_scores

def example_accuracy(y_true, y_pred):
    """
    计算多标签分类的示例准确率

    参数:
    y_true (numpy.ndarray): 真实标签，形状为 (n_samples, n_labels)
    y_pred (numpy.ndarray): 预测标签，形状为 (n_samples, n_labels)

    返回:
    float: 示例准确率
    """
    intersection = np.logical_and(y_true, y_pred).sum(axis=1)
    union = np.logical_or(y_true, y_pred).sum(axis=1)
    example_accuracy = np.mean(intersection / union)
    return example_accuracy

def metrics(gt,pred,threshold,n_classes,weights_file,classes,normal_class,task):
    # 数据类型为np。其中gt真实标签，pred预测结果（最后一层的logits）
    # gt: (n_samples, n_classes)
    # pred: (n_samples, n_classes)
    # pred_np: (n_samples, n_classes)
    AUROCs,metrics = compute_AUCs(gt, pred, n_class=n_classes)
    AUROC_avg = metrics['mean_auc']
    # 计算每个标签的AUPRC
    auprc_scores = [average_precision_score(gt[:, i], pred[:, i]) for i in range(gt.shape[1])]
    # 计算宏平均AUPRC
    macro_auprc = np.mean(auprc_scores)
    # 看看F1，acc
    # pred_np就是经过阈值转换的结果

    pred_np = round_compute(pred,threshold,n_classes)
    F1s,F1_macro = compute_F1s_threshold(gt, pred, threshold, n_class=n_classes)
    Accs,acc_mean,acc_sample = compute_Accs_threshold(gt, pred, threshold, n_class=n_classes)
    acc_exmaple = example_accuracy(gt, pred_np)
    h_loss = hamming_loss(gt,pred_np)
    j_score = jaccard_score(gt, pred_np,average='macro')
    if task == 'cinc':
        challenge_metric = compute_challenge_metric(load_weights(weights_file, classes),
                                                    gt, pred_np, classes, normal_class)
    else:
        challenge_metric = None
    # f1_modified = calculate_f1(compute_modified_confusion_matrix(gt, pred_np))  # FIXME: 这个modified结果有点低，看看为啥
    # f1_scratch = calculate_f1(confusion_m(gt, pred_np))

    metrics_sspn = {'sensitivity': [], 'specificity': [], 'PPV': [], 'NPV': []}
    for i in range(gt.shape[1]):  # 遍历每个类别
        TP = np.sum((gt[:, i] == 1) & (pred_np[:, i] == 1))
        TN = np.sum((gt[:, i] == 0) & (pred_np[:, i] == 0))
        FP = np.sum((gt[:, i] == 0) & (pred_np[:, i] == 1))
        FN = np.sum((gt[:, i] == 1) & (pred_np[:, i] == 0))

        sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
        specificity = TN / (TN + FP) if TN + FP > 0 else 0
        PPV = TP / (TP + FP) if TP + FP > 0 else 0
        NPV = TN / (TN + FN) if TN + FN > 0 else 0

        metrics_sspn['sensitivity'].append(sensitivity)
        metrics_sspn['specificity'].append(specificity)
        metrics_sspn['PPV'].append(PPV)
        metrics_sspn['NPV'].append(NPV)
    macro_avg_sensitivity = np.mean(metrics_sspn['sensitivity'])
    macro_avg_specificity = np.mean(metrics_sspn['specificity'])
    macro_avg_PPV = np.mean(metrics_sspn['PPV'])
    macro_avg_NPV = np.mean(metrics_sspn['NPV'])
    # 计算多标签混淆矩阵
    mcm = multilabel_confusion_matrix(gt, pred_np)

    # 计算总的TP, FP, TN, FN
    TP_mcm = mcm[:, 1, 1].sum()
    FP_mcm = mcm[:, 0, 1].sum()
    TN_mcm = mcm[:, 0, 0].sum()
    FN_mcm = mcm[:, 1, 0].sum()

    # 计算微平均
    micro_avg_sensitivity = TP_mcm / (TP_mcm + FN_mcm) if TP_mcm + FN_mcm > 0 else 0
    micro_avg_specificity = TN_mcm / (TN_mcm + FP_mcm) if TN_mcm + FP_mcm > 0 else 0
    micro_avg_PPV = TP_mcm / (TP_mcm + FP_mcm) if TP_mcm + FP_mcm > 0 else 0
    micro_avg_NPV = TN_mcm / (TN_mcm + FN_mcm) if TN_mcm + FN_mcm > 0 else 0
    micro_f1 = f1_score(gt, pred_np, average='micro')


    metrics_dict = {
    'Accuracy_element': acc_mean,
    'Acc_sample':acc_sample,
    'Acc_example':acc_exmaple,
    'F1_score': F1_macro,
    'Hamming_loss': h_loss,
    'Jaccard_score':j_score,
    'AUROC_avg':AUROC_avg,
    'Macro_AUPRC':macro_auprc,
    'Challenge_metric':challenge_metric,
    'Macro_avg_sensitivity': macro_avg_sensitivity,
    'Macro_avg_specificity': macro_avg_specificity,
    'Macro_avg_PPV': macro_avg_PPV,
    'Macro_avg_NPV': macro_avg_NPV,
    'Micro_F1': micro_f1,
    'Micro_avg_sensitivity': micro_avg_sensitivity,
    'Micro_avg_specificity': micro_avg_specificity,
    'Micro_avg_PPV': micro_avg_PPV,
    'Micro_avg_NPV': micro_avg_NPV,

    }
    AUC_scores = pd.DataFrame(AUROCs,columns=['AUCs_scores'])
    F1_scores = pd.DataFrame(F1s,columns=['F1_scores'])
    Acc_socres = pd.DataFrame(Accs,columns=['Acc_scores'])
    sensitivity_scores = pd.DataFrame(metrics_sspn['sensitivity'],columns=['sensitivity_scores'])
    specificity_scores = pd.DataFrame(metrics_sspn['specificity'],columns=['specificity_scores'])
    PPV_scores = pd.DataFrame(metrics_sspn['PPV'],columns=['PPV_scores'])
    NPV_scores = pd.DataFrame(metrics_sspn['NPV'],columns=['NPV_scores'])

    # F1_scores_scratch = pd.DataFrame(f1_scratch,columns=['F1_scores_scratch'])
    # F1_modified_scores = pd.DataFrame(f1_modified,columns=['F1_modified_scores'])

    df_scores = pd.concat([AUC_scores,F1_scores,Acc_socres,sensitivity_scores,specificity_scores,PPV_scores,NPV_scores], axis=1)
    print(metrics_dict)
    return metrics_dict,df_scores,pred_np

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples


def generate_results(idxs, y_true, y_pred_logits,y_pred,metrics='all'):
    return evaluate_experiment(y_true[idxs], y_pred_logits[idxs],y_pred[idxs],metrics='all')


def evaluate_experiment(y_true, y_pred_logits, y_pred,metrics='all'):
    results = {}

    # label based metric
    if metrics == 'all':
        results['macro_auc'] = roc_auc_score(y_true, y_pred_logits, average='macro')
        results['macro_prc'] = average_precision_score(y_true, y_pred_logits, average='macro')
        Accs = []
        for i in range(y_true.shape[1]):
            Accs.append(accuracy_score(y_true[:, i], y_pred[:, i]))
        acc_mean = np.mean(np.array(Accs))
        results['acc_element'] = acc_mean
        results['acc_sample'] = accuracy_score(y_true, y_pred)
        results['acc_example'] = example_accuracy(y_true, y_pred)
        results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics_sspn = {'sensitivity': [], 'specificity': [], 'PPV': [], 'NPV': []}
        for i in range(y_true.shape[1]):  # 遍历每个类别
            TP = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            TN = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            FP = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            FN = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            specificity = TN / (TN + FP) if TN + FP > 0 else 0
            PPV = TP / (TP + FP) if TP + FP > 0 else 0
            NPV = TN / (TN + FN) if TN + FN > 0 else 0

            metrics_sspn['sensitivity'].append(sensitivity)
            metrics_sspn['specificity'].append(specificity)
            metrics_sspn['PPV'].append(PPV)
            metrics_sspn['NPV'].append(NPV)
        macro_avg_sensitivity = np.mean(metrics_sspn['sensitivity'])
        macro_avg_specificity = np.mean(metrics_sspn['specificity'])
        macro_avg_PPV = np.mean(metrics_sspn['PPV'])
        macro_avg_NPV = np.mean(metrics_sspn['NPV'])

        results['macro_sensitivity'] = macro_avg_sensitivity
        results['macro_specificity'] = macro_avg_specificity
        results['macro_PPV'] = macro_avg_PPV
        results['macro_NPV'] = macro_avg_NPV

        results['hamming_score'] = 1-hamming_loss(y_true, y_pred)
        results['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro')
        # 计算多标签混淆矩阵
        mcm = multilabel_confusion_matrix(y_true, y_pred)

        # 计算总的TP, FP, TN, FN
        TP_mcm = mcm[:, 1, 1].sum()
        FP_mcm = mcm[:, 0, 1].sum()
        TN_mcm = mcm[:, 0, 0].sum()
        FN_mcm = mcm[:, 1, 0].sum()

        # 计算微平均
        micro_avg_sensitivity = TP_mcm / (TP_mcm + FN_mcm) if TP_mcm + FN_mcm > 0 else 0
        micro_avg_specificity = TN_mcm / (TN_mcm + FP_mcm) if TN_mcm + FP_mcm > 0 else 0
        micro_avg_PPV = TP_mcm / (TP_mcm + FP_mcm) if TP_mcm + FP_mcm > 0 else 0
        micro_avg_NPV = TN_mcm / (TN_mcm + FN_mcm) if TN_mcm + FN_mcm > 0 else 0
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        results['micro_f1'] = micro_f1
        results['micro_sensitivity'] = micro_avg_sensitivity
        results['micro_specificity'] = micro_avg_specificity
        results['micro_PPV'] = micro_avg_PPV
        results['micro_NPV'] = micro_avg_NPV
    elif metrics == 'macro_auc':
        results['macro_auc'] = roc_auc_score(y_true, y_pred_logits, average='macro')
    elif metrics == 'marco_prc':
        results['macro_prc'] = average_precision_score(y_true, y_pred_logits, average='macro')
    elif metrics == 'acc_element':
        Accs = []
        for i in range(y_true.shape[1]):
            Accs.append(accuracy_score(y_true[:, i], y_pred[:, i]))
        acc_mean = np.mean(np.array(Accs))
        results['acc_element'] = acc_mean
    elif metrics == 'acc_sample':
        results['acc_sample'] = accuracy_score(y_true, y_pred)
    elif metrics == 'macro_f1':
        results['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    elif metrics == 'hamming_score':
        results['hamming_score'] = 1-hamming_loss(y_true, y_pred)
    elif metrics == 'jaccard_score':
        results['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro')
    else:
        metrics_sspn = {'sensitivity': [], 'specificity': [], 'PPV': [], 'NPV': []}
        for i in range(y_true.shape[1]):  # 遍历每个类别
            TP = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            TN = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            FP = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            FN = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

            sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
            specificity = TN / (TN + FP) if TN + FP > 0 else 0
            PPV = TP / (TP + FP) if TP + FP > 0 else 0
            NPV = TN / (TN + FN) if TN + FN > 0 else 0

            metrics_sspn['sensitivity'].append(sensitivity)
            metrics_sspn['specificity'].append(specificity)
            metrics_sspn['PPV'].append(PPV)
            metrics_sspn['NPV'].append(NPV)
        macro_avg_sensitivity = np.mean(metrics_sspn['sensitivity'])
        macro_avg_specificity = np.mean(metrics_sspn['specificity'])
        macro_avg_PPV = np.mean(metrics_sspn['PPV'])
        macro_avg_NPV = np.mean(metrics_sspn['NPV'])

        results['macro_sensitivity'] = macro_avg_sensitivity
        results['macro_specificity'] = macro_avg_specificity
        results['macro_PPV'] = macro_avg_PPV
        results['macro_NPV'] = macro_avg_NPV



    df_result = pd.DataFrame(results, index=[0])
    if metrics == 'all':
        return df_result
    else:
        return df_result[metrics]

if __name__ == '__main__':
    gt = torch.tensor(np.load(r"D:\Filez\research\experiment\ptbxl_multi_label\output\all\20230727151152\y_test.npy"))
    pred = torch.tensor(np.load(r"D:\Filez\research\experiment\ptbxl_multi_label\output\all\20230727151152\test_logits.npy"))
    metrics(gt,pred,71)