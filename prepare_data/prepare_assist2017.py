from sklearn.model_selection import train_test_split, KFold
from EduData import get_data
import numpy as np
import pandas as pd
import tqdm
import os

path = './data/anonymized_full_release_competition_dataset'

if not os.path.exists(path + '/anonymized_full_release_competition_dataset.csv'):
  get_data("assistment-2017", "./data")

data = pd.read_csv(
  path + '/anonymized_full_release_competition_dataset.csv',
  usecols=['startTime', 'endTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'correct']
).dropna(subset=['skill', 'problemId'])

data.timeTaken = data.timeTaken.astype(int)

skills = data.skill.unique().tolist()
problems = data.problemId.unique().tolist()
at = data.timeTaken.unique()
user_seqs = [u.sort_values('endTime') for _, u in list(data.groupby('studentId'))]

# question id from 1 to #num_skill
skill2id = {p: i + 1 for i, p in enumerate(skills)}
problem2id = {p: i + 1 for i, p in enumerate(problems)}
at2id = {a: i for i, a in enumerate(at)}

it = set()
avg_it = np.array([])
# calculate interval time
for i, seq in enumerate(user_seqs):
  seq = seq.copy()
  items = seq.endTime.diff(1) // 60
  items.iloc[0] = 0
  items = items.astype(int)
  items[items > 43200] = 43200
  seq['it'] = items
  user_seqs[i] = seq
  for item in items.unique():
    it.add(item)

it2id = {a: i for i, a in enumerate(it)}

id2it = {id: it for it, id in it2id.items()}
with open(path + '/id2it', 'w', encoding='utf-8') as f:
  f.write(str(id2it))

if not os.path.exists(path + '/id2skill'):
  id2skill = {}
  for skill, id in skill2id.items():
    id2skill[id] = skill
  with open(path + '/id2skill', 'w', encoding='utf-8') as f:
    f.write(str(id2skill))

# problems to skills
problem2skill = {}
for s, p in zip(data.skill.to_numpy(), data.problemId.to_numpy()):
  problem2skill[problem2id[p]] = skill2id[s]
with open(path + '/problem2skill', 'w', encoding='utf-8') as f:
  f.write(str(problem2skill))


def parse_all_seq(students):
  all_sequences = []
  for seq in tqdm.tqdm(students, 'parse student sequence:\t'):
    student_sequence = parse_student_seq(seq)
    all_sequences.extend([student_sequence])
  return all_sequences


def parse_student_seq(student):
  seq = student
  s = [skill2id[q] for q in seq.skill.tolist()]
  a = seq.correct.tolist()
  p = [problem2id[p] for p in seq.problemId.tolist()]
  it = [it2id[int(x)] for x in seq.it.tolist()]
  at = [at2id[int(x)] for x in seq.timeTaken.tolist()]
  return s, a, p, it, at


sequences = parse_all_seq(user_seqs)

# split train data and test data
train_data, test_data = train_test_split(sequences, test_size=.2, random_state=5)
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)


def sequences2l(sequences, trg_path):
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


# split into 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=5)
idx = 0
for train_data_1, valid_data in kfold.split(train_data):
  sequences2l(train_data[train_data_1], path + '/train' + str(idx) + '.txt')
  sequences2l(train_data[valid_data], path + '/valid' + str(idx) + '.txt')
  idx += 1

sequences2l(test_data, path + '/test.txt')
