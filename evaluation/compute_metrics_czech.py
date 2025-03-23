import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def compute_metrics_czech(eval_preds):

    print('starting compute metrics')
    labels = eval_preds.label_ids
    scores = eval_preds.predictions

    # labels = np.any(labels, axis=1).astype(np.int32)
    scores = sigmoid(scores)
    assert labels.shape == scores.shape

    roc_auc = roc_auc_score(labels, scores)
    PR_auc = average_precision_score(labels, scores)
    print('computed aucs')
    scores = np.array(scores>0.5).astype(np.int32)
    confusion_mat = confusion_matrix(labels, scores)

    tp, fp = confusion_mat[1,1], confusion_mat[0,1]
    tn, fn = confusion_mat[0,0], confusion_mat[1,0]

    accuracy = accuracy_score(labels, scores)
    accuracy_mean = balanced_accuracy_score(labels, scores)
    f1_churn = f1_score(labels, scores, average='binary')
    f1_mean = f1_score(labels, scores, average='macro')
    print('finished compute metrics')
    return {
        'ROC_AUC': roc_auc,
        'PrecRec_Auc': PR_auc,
        'accuracy_global': accuracy,
        'accuracy_mean_class': accuracy_mean,
        'F1_churn': f1_churn,
        'F1_mean': f1_mean,
        'true_pos': tp,
        'false_pos': fp,
        'false_neg': fn,
        'true_neg': tn,
    }
