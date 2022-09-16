import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr


def binary_entropy(target, pred):
  loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
  return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
  return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
  all_pred = all_pred.copy()
  all_pred[all_pred > 0.5] = 1.0
  all_pred[all_pred <= 0.5] = 0.0
  return metrics.accuracy_score(all_target, all_pred)


def compute_rmse(all_target, all_pred):
  return np.sqrt(metrics.mean_squared_error(all_target, all_pred))


def compute_r2(all_target, all_pred):
  return np.power(pearsonr(all_target, all_pred)[0], 2)
