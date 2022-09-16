import random
import sys

import torch
import numpy as np

from model import OKT
from load_data import DATA

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
  dataset = sys.argv[1] if len(sys.argv) > 1 else 'assist2017'

  if dataset not in ['assist2017', 'assist2012', 'nips2020_1_2']:
    raise Exception(dataset + ' is not target dataset')

  n_at, n_it, n_exercise, n_question = 0, 0, 0, 0
  has_at = True

  if dataset == 'assist2017':
    n_at = 1326
    n_it = 2873
    n_exercise = 3162
    n_question = 102
    seq_len = 500
    dataset_path = 'anonymized_full_release_competition_dataset'
  elif dataset == 'assist2012':
    n_at = 26411
    n_it = 29694
    n_exercise = 53091
    n_question = 265
    seq_len = 100
    dataset_path = '2012-2013-data-with-predictions-4-final'
  elif dataset == 'nips2020_1_2':
    n_it = 42148
    n_exercise = 27613
    n_question = 1125
    seq_len = 100
    dataset_path = 'NIPS2020/task_1_2'
    has_at = False
  else:
    raise Exception('no dataset named %s' % dataset)

  d_q, d_e = 32, 128
  d_p, d_a = 128, 128
  d_at = 50
  d_h = 128

  dropout = 0.3

  if dataset == 'assist2012':
    batch_size = 128
    lr = 1e-3
    lr_decay_step = 10
    lr_decay_rate = 0.1
    epoch = 10
  elif dataset == 'assist2017':
    batch_size = 64
    lr = 3e-3
    lr_decay_step = 10
    lr_decay_rate = 0.5
    epoch = 30
  elif dataset == 'nips2020_1_2':
    batch_size = 512
    lr = 2e-3
    lr_decay_step = 5
    lr_decay_rate = 0.5
    epoch = 20

  data_path = './data/' + dataset_path
  dat = DATA(seqlen=seq_len, separate_char=',', has_at=has_at)
  test_data = dat.load_data(data_path + '/test.txt')
  model_file_path = '.okt-' + dataset + '.params'
  # k-fold cross validation
  k, test_r2_sum, test_auc_sum, test_accuracy_sum = 5, .0, .0, .0
  for i in range(k):
    okt = OKT(n_at, n_it, n_exercise, n_question,
          d_e, d_q, d_a, d_at, d_p, d_h, batch_size=batch_size, dropout=dropout)
    train_data = dat.load_data(data_path + '/train' + str(i) + '.txt')
    valid_data = dat.load_data(data_path + '/valid' + str(i) + '.txt')
    best_train_auc, best_valid_auc = okt.train(train_data, valid_data,
                           epoch=epoch, lr=lr, lr_decay_step=lr_decay_step,
                           lr_decay_rate=lr_decay_rate,
                           filepath=model_file_path)
    print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    # test
    okt.load(model_file_path)
    _, (_, test_r2, test_auc, test_accuracy) = okt.eval(test_data)
    print("[fold %d] r2: %.6f, auc: %.6f, accuracy: %.6f" % (i, test_r2, test_auc, test_accuracy))
    test_r2_sum += test_r2
    test_auc_sum += test_auc
    test_accuracy_sum += test_accuracy
  print('%d-fold validation:' % k)
  print('avg of test data (r2, auc, accuracy): %f, %f, %f' % (
    test_r2_sum / k, test_auc_sum / k, test_accuracy_sum / k))
