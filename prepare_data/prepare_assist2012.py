from EduData import get_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tqdm
import os

path = './data/2012-2013-data-with-predictions-4-final'

if not os.path.exists(path + '/2012-2013-data-with-predictions-4-final.csv'):
  get_data('assistment-2012-2013-non-skill', './data')

data = pd.read_csv(
  path + '/2012-2013-data-with-predictions-4-final.csv',
  usecols=['start_time', 'end_time', 'overlap_time', 'user_id', 'skill_id', 'skill', 'problem_id', 'correct']
)
data = data.dropna(subset=['skill_id', 'problem_id'])
data.start_time = pd.to_datetime(data.start_time).astype(int)
data.end_time = pd.to_datetime(data.end_time).astype(int)
data['overlap_time'] = ((data.end_time - data.start_time) // 1e9).astype(int)
data = data[data.overlap_time >= 0]
data = data.sort_values('end_time')
data.correct = data.correct.astype(int)

user_seqs = [u.reset_index(drop=True) for _, u in list(data.groupby('user_id'))]

it_set = set()
# calculate interval time
for i, seq in enumerate(user_seqs):
  seq = seq.copy()
  items = seq.end_time.diff(1) // (1e9 * 60)
  items[0] = 0
  items = items.astype(int)
  items[items > 43200] = 43200
  seq['it'] = items
  user_seqs[i] = seq
  for item in items.unique():
      it_set.add(item)

skills = data.skill_id.unique().tolist()
problems = data.problem_id.unique().tolist()
at = data.overlap_time.unique()
# question id from 1 to #num_skill
skill2id = {p: i + 1 for i, p in enumerate(skills)}
problem2id = {p: i + 1 for i, p in enumerate(problems)}
at2id = {a: i for i, a in enumerate(at)}
it2id = {a: i for i, a in enumerate(it_set)}

# problems to skills
if not os.path.exists(path + '/problem2skill'):
  problem2skill = {}
  for s, p in zip(np.array(data.skill_id), np.array(data.problem_id)):
    problem2skill[problem2id[p]] = skill2id[s]
  with open(path + '/problem2skill', 'w', encoding='utf-8') as f:
    f.write(str(problem2skill))

if not os.path.exists(path + '/id2skill'):
  id2skill = {}
  for skill, id in skill2id.items():
    id2skill[id] = skill
  with open(path + '/id2skill', 'w', encoding='utf-8') as f:
    f.write(str(id2skill))

if not os.path.exists(path + '/real_id2skill_name'):
  real_id2skill_name = {}
  for id, name in zip(np.array(data.skill_id), np.array(data.skill)):
    real_id2skill_name[id] = name
  with open(path + '/real_id2skill_name', 'w', encoding='utf-8') as f:
    f.write(str(real_id2skill_name))

if not os.path.exists(path + '/id2it'):
  id2it = {}
  for it, id in it2id.items():
    id2it[id] = it
  with open(path + '/id2it', 'w', encoding='utf-8') as f:
    f.write(str(id2it))


def parse_all_seq(students):
  all_sequences = []
  for student in tqdm.tqdm(students, 'parse student sequence:\t'):
    student_sequence = parse_student_seq(student)
    all_sequences.extend([student_sequence])
  return all_sequences


def parse_student_seq(student):
  seq = student
  s = [skill2id[q] for q in seq.skill_id.tolist()]
  a = seq.correct.tolist()
  p = [problem2id[p] for p in seq.problem_id.tolist()]
  it = [it2id[int(x)] for x in seq.it.tolist()]
  at = [at2id[int(x)] for x in seq.overlap_time.tolist()]
  return s, a, p, it, at


def sequences2l(sequences, trg_path):
  if not os.path.exists(trg_path):
    with open(trg_path, 'a', encoding='utf8') as f:
      for seq in tqdm.tqdm(sequences, 'write data into file %s' % trg_path):
        s_seq, a_seq, p_seq, it_seq, at_seq = seq
        seq_len = len(s_seq)
        f.write(str(seq_len) + '\n')
        f.write(','.join([str(s) for s in s_seq]) + '\n')
        f.write(','.join([str(a) for a in a_seq]) + '\n')
        f.write(','.join([str(p) for p in p_seq]) + '\n')
        f.write(','.join([str(i) for i in it_seq]) + '\n')
        f.write(','.join([str(a) for a in at_seq]) + '\n')


sequences = parse_all_seq(user_seqs)

# split train data and test data
train_data, test_data = train_test_split(sequences, test_size=.2, random_state=10)
train_data = np.array(train_data)
test_data = np.array(test_data)


# split into 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=10)
idx = 0
for train_data_1, valid_data in kfold.split(train_data):
  sequences2l(train_data[train_data_1], path + '/train' + str(idx) + '.txt')
  sequences2l(train_data[valid_data], path + '/valid' + str(idx) + '.txt')
  idx += 1

sequences2l(test_data, path + '/test.txt')
