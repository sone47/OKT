import math
import logging
import torch
import torch.nn as nn
import numpy as np
import tqdm

from .OKTNet import OKTNet
from .OKTNOTNet import OKTNet as OKTNOTNet
from .OKTNUNet import OKTNet as OKTNUNet
from .OKTNENet import OKTNet as OKTNENet
from .utils import binary_entropy, compute_auc, compute_accuracy, compute_r2, compute_rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(net, optimizer, criterion, batch_size, q_data, a_data, e_data, it_data, at_data=None):
  if at_data is None:
    at_data = []
  net.train()
  n = int(math.ceil(len(e_data) / batch_size))
  shuffled_ind = np.arange(e_data.shape[0])
  np.random.shuffle(shuffled_ind)
  e_data = e_data[shuffled_ind]
  at_data = at_data[shuffled_ind]
  a_data = a_data[shuffled_ind]
  it_data = it_data[shuffled_ind]

  pred_list = []
  target_list = []
  for idx in tqdm.tqdm(range(n), 'Training'):
    optimizer.zero_grad()

    q_one_seq = q_data[idx * batch_size: (idx + 1) * batch_size, :]
    e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
    at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
    a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
    it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

    input_q = torch.from_numpy(q_one_seq).long().to(device)
    input_e = torch.from_numpy(e_one_seq).long().to(device)
    input_a = torch.from_numpy(a_one_seq).long().to(device)
    input_at = torch.from_numpy(at_one_seq).long().to(device)
    input_it = torch.from_numpy(it_one_seq).long().to(device)
    target = torch.from_numpy(a_one_seq).float().to(device)

    pred = net(input_q, input_a, input_e, input_it, input_at)

    mask = input_e[:, 1:] > 0
    masked_pred = pred[:, 1:][mask]
    masked_truth = target[:, 1:][mask]

    loss = criterion(masked_pred, masked_truth)

    loss.backward()

    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
    optimizer.step()

    masked_pred = masked_pred.detach().cpu().numpy()
    masked_truth = masked_truth.detach().cpu().numpy()
    pred_list.append(masked_pred)
    target_list.append(masked_truth)

  all_pred = np.concatenate(pred_list, axis=0)
  all_target = np.concatenate(target_list, axis=0)

  loss = binary_entropy(all_target, all_pred)
  r2 = compute_r2(all_target, all_pred)
  auc = compute_auc(all_target, all_pred)
  accuracy = compute_accuracy(all_target, all_pred)

  return loss, r2, auc, accuracy


def test_one_epoch(net, batch_size, q_data, a_data, e_data, it_data, at_data=None, return_it=False):
  if at_data is None:
    at_data = []
  net.eval()
  n = int(math.ceil(len(e_data) / batch_size))

  pred_list = []
  target_list = []
  it_list = []
  mask_list = []

  for idx in tqdm.tqdm(range(n), 'Testing'):
    q_one_seq = q_data[idx * batch_size: (idx + 1) * batch_size, :]
    e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
    at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
    a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
    it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]

    input_q = torch.from_numpy(q_one_seq).long().to(device)
    input_e = torch.from_numpy(e_one_seq).long().to(device)
    input_a = torch.from_numpy(a_one_seq).long().to(device)
    input_at = torch.from_numpy(at_one_seq).long().to(device)
    input_it = torch.from_numpy(it_one_seq).long().to(device)
    target = torch.from_numpy(a_one_seq).float().to(device)

    with torch.no_grad():
      pred = net(input_q, input_a, input_e, input_it, input_at)

      mask = input_e[:, 1:] > 0
      masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
      masked_truth = target[:, 1:][mask].detach().cpu().numpy()
      if return_it:
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        it = torch.from_numpy(it_one_seq).long().to(device)
        masked_it = it[:, 1:][mask].detach().cpu().numpy()
        it_list.append(masked_it)

      pred_list.append(masked_pred)
      target_list.append(masked_truth)
      mask_list.append(mask.long().cpu().numpy())

  all_pred = np.concatenate(pred_list, axis=0)
  all_target = np.concatenate(target_list, axis=0)
  mask_list = np.concatenate(mask_list, axis=0)

  loss = binary_entropy(all_target, all_pred)
  r2 = compute_r2(all_target, all_pred)
  auc = compute_auc(all_target, all_pred)
  accuracy = compute_accuracy(all_target, all_pred)
  rmse = compute_rmse(all_target, all_pred)

  if return_it:
    all_it = np.concatenate(it_list, axis=0)
    return (all_pred, all_target, all_it, mask_list), (loss, rmse, r2, auc, accuracy)
  else:
    return (all_pred, all_target, mask_list), (loss, rmse, r2, auc, accuracy)


net_dict = {
  'okt': OKTNet,
  'okt_not': OKTNOTNet,
  'okt_nu': OKTNUNet,
  'okt_ne': OKTNENet
}

class OKT:
  def __init__(self, n_at, n_it, n_exercise, n_question, d_e, d_q, d_a, d_at, d_p, d_h, batch_size=64, dropout=0.2, net_type='okt'):
    super(OKT, self).__init__()

    if net_type not in net_dict:
      raise Exception('Parameter model \'' + net_type + '\' is wrong.')

    net = net_dict[net_type]

    self.okt_net = net(n_question, n_exercise, n_it, n_at, d_e, d_q, d_a, d_at, d_p, d_h,
                      dropout=dropout, device=device).to(device)
    self.batch_size = batch_size

  def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5,
        filepath=None) -> ...:
    optimizer = torch.optim.Adam(self.okt_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
    criterion = nn.BCELoss()
    best_train_auc, best_test_auc = .0, .0

    for idx in range(epoch):
      train_loss, train_r2, train_auc, train_accuracy = train_one_epoch(self.okt_net, optimizer, criterion,
                                        self.batch_size, *train_data)
      print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
      if train_auc > best_train_auc:
        best_train_auc = train_auc

      if test_data is not None:
        _, (_, _, test_r2, test_auc, test_accuracy) = self.eval(test_data)
        print("[Epoch %d] r2: %.6f, auc: %.6f, accuracy: %.6f" % (idx, test_r2, test_auc, test_accuracy))
        scheduler.step()
        if test_auc > best_test_auc:
          best_test_auc = test_auc
          if filepath is not None:
            self.save(filepath)

    return best_train_auc, best_test_auc

  def eval(self, test_data, return_it=False) -> ...:
    return test_one_epoch(self.okt_net, self.batch_size, *test_data, return_it)

  def save(self, filepath) -> ...:
    torch.save(self.okt_net.state_dict(), filepath)
    logging.info("save parameters to %s" % filepath)

  def load(self, filepath) -> ...:
    self.okt_net.load_state_dict(torch.load(filepath, map_location='cpu'))
    logging.info("load parameters from %s" % filepath)
