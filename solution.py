import py_entitymatching as em
from os.path import join

# 1. Reading Data
left = em.read_csv_metadata(join('data', "ltable.csv"), key='id')
right = em.read_csv_metadata(join('data', "rtable.csv"), key='id')
train = em.read_csv_metadata(join('data', "train.csv"))
train['id'] = range(0, len(train))
em.set_key(A, 'id')
