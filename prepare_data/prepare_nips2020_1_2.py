from EduData import get_data
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import os

path = './data/NIPS2020/public_data'
data_path = path + '/data.csv'
task = 'task_1_2'

if not os.path.exists(path):
  get_data('NIPS-2020', './data')


# load task data
if not os.path.exists(data_path):
  data = pd.read_csv(path + '/train_data/train_' + task + '.csv', usecols=['QuestionId', 'UserId', 'AnswerId', 'IsCorrect'])

new_path = './data/NIPS2020/' + task

# load answer_data: for interval time
aid2time_path = new_path + '/aid2time'
if not os.path.exists(aid2time_path):
  print('load aid2time data...')
  answer_data = pd.read_csv(path + '/metadata/answer_metadata_' + task + '.csv', usecols=['AnswerId', 'DateAnswered'])
  answer_data = answer_data.dropna()
  answer_data.AnswerId = answer_data.AnswerId.astype(int)
  answer_data.DateAnswered = pd.to_datetime(answer_data.DateAnswered).astype(int) // (1e9 * 60)
  aid2time = {}
  for _, row in tqdm(answer_data.iterrows()):
    aid2time[row['AnswerId']] = row['DateAnswered']
  with open(aid2time_path, 'w', encoding='utf-8') as f:
    f.write(str(aid2time))
else:
  with open(aid2time_path, 'r', encoding='utf-8') as f:
    for line in f:
      aid2time = eval(line)

# load question data: for skill
qid2skill_path = new_path + '/qid2skill'
if not os.path.exists(qid2skill_path):
  print('load qid2skill data...')
  # QuestionId(int), SubjectId(object)
  skill_data = pd.read_csv(path + '/metadata/question_metadata_' + task + '.csv')
  skill_data = skill_data.dropna()
  qid2s = {}
  for _, row in tqdm(skill_data.iterrows()):
    qid2s[row['QuestionId']] = row['SubjectId']
  with open(qid2skill_path, 'w', encoding='utf-8') as f:
    f.write(str(qid2s))
else:
  with open(qid2skill_path, 'r', encoding='utf-8') as f:
    for line in f:
      qid2s = eval(line)


# complete data
data_path = new_path + '/data.csv'
if not os.path.exists(data_path):
  answer_time = pd.Series([0] * len(data))
  skills = pd.Series([''] * len(data))
  for index, row in data.iterrows():
    answer_time[index] = aid2time[row['AnswerId']]
    skills[index] = qid2s[row['QuestionId']]
  data['DateAnswered'] = answer_time
  data['SubjectId'] = skills

  data.to_csv(data_path)
else:
  data = pd.read_csv(data_path)
data = data.dropna().drop_duplicates(subset=['DateAnswered', 'UserId']).sort_values('DateAnswered')


# analysis
if not os.path.exists(new_path + '/skill2id'):
  skills = set()
  for s in data.SubjectId:
    skills.add(s)
  skill2id = {p: i + 1 for i, p in enumerate(skills)}
  with open(new_path + '/skill2id', 'w', encoding='utf-8') as f:
    f.write(str(skill2id))
else:
  with open(new_path + '/skill2id', 'r', encoding='utf-8') as f:
    for line in f:
      skill2id = eval(line)

if not os.path.exists(new_path + '/problem2id'):
  problems = data.QuestionId.unique().tolist()
  problem2id = {p: i + 1 for i, p in enumerate(problems)}
  with open(new_path + '/problem2id', 'w', encoding='utf-8') as f:
    f.write(str(problem2id))
else:
  with open(new_path + '/problem2id', 'r', encoding='utf-8') as f:
    for line in f:
      problem2id = eval(line)

users = data.UserId.unique()

print("number of skills: %d" % len(skill2id))
print("number of problems: %d" % len(problem2id))
print("number of users: %d" % len(users))

avg_length = 0
for u in users:
  avg_length += len(data[data.UserId == u])
avg_length /= len(users)
print('avg length of sequence: %.2f' % avg_length)


if not os.path.exists(new_path + '/it2id'):
  it = set()
  avg_it = np.array([])
  # calculate interval time
  for u in users:
    startTime = data[data.UserId == u].DateAnswered
    items = startTime.diff()
    items.iloc[0] = 0
    avg_it = np.concatenate((avg_it, [items.mean()]))
    items[items > 43200] = 43200
    for item in items.unique():
      it.add(item)

  it2id = {a: i for i, a in enumerate(it)}
  print('Avg.minutes per interval time: %.2f' % avg_it.mean())
  with open(new_path + '/it2id', 'w', encoding='utf-8') as f:
    f.write(str(it2id))
else:
  with open(new_path + '/it2id', 'r', encoding='utf-8') as f:
    for line in f:
      it2id = eval(line)
print("number of interval time: %d" % len(it2id))


if not os.path.exists(new_path + '/problem2skill'):
  problem2skill = {}
  for ss, p in zip(np.array(data.SubjectId), np.array(data.QuestionId)):
    problem2skill[problem2id[p]] = skill2id[ss]
  with open(new_path + '/problem2skill', 'w', encoding='utf-8') as f:
    f.write(str(problem2skill))
else:
  with open(new_path + '/problem2skill', 'r', encoding='utf-8') as f:
    for line in f:
      problem2skill = eval(line)


def parse_all_seq(students):
  all_sequences = []
  for student_id in tqdm(students, 'parse student sequence'):
    student_sequence = parse_student_seq(data[data.UserId == student_id])
    all_sequences.extend([student_sequence])
  return all_sequences


def parse_student_seq(student):
  seq = student
  s = [problem2skill[problem2id[p]] for p in seq.QuestionId.tolist()]
  a = seq.IsCorrect.tolist()
  p = [problem2id[p] for p in seq.QuestionId.tolist()]
  it = [0]
  startTime = np.array(seq.DateAnswered)
  for i in range(1, len(startTime)):
    item = startTime[i] - startTime[i - 1]
    if item > 43200:
      item = 43200
    it.append(it2id[item])
  return s, a, p, it


sequences = parse_all_seq(users)

# split train data and test data
train_data, test_data = train_test_split(sequences, test_size=.2, random_state=10)
train_data = np.array(train_data)
test_data = np.array(test_data)


def sequences2l(sequences, trg_path, separate_char=','):
  with open(trg_path, 'a', encoding='utf8') as f:
    for seq in tqdm(sequences, 'write data into file %s' % trg_path):
      s_seq, a_seq, p_seq, it_seq = seq
      seq_len = len(s_seq)
      f.write(str(seq_len) + '\n')
      f.write(separate_char.join([str(s) for s in s_seq]) + '\n')
      f.write(separate_char.join([str(a) for a in a_seq]) + '\n')
      f.write(separate_char.join([str(p) for p in p_seq]) + '\n')
      f.write(separate_char.join([str(i) for i in it_seq]) + '\n')


# split into 5 folds
kfold = KFold(n_splits=5, shuffle=True, random_state=10)
idx = 0
for train_data_1, valid_data in kfold.split(train_data):
  sequences2l(train_data[train_data_1], new_path + '/train' + str(idx) + '.txt')
  sequences2l(train_data[valid_data], new_path + '/valid' + str(idx) + '.txt')
  idx += 1

sequences2l(test_data, new_path + '/test.txt')
