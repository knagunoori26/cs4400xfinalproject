import py_entitymatching as em
from os.path import join
import numpy as np
import pandas as pd
import distance as distance

# 1. Reading Data
left = em.read_csv_metadata(join('data', "ltable.csv"), key='id')
right = em.read_csv_metadata(join('data', "rtable.csv"), key='id')
train = em.read_csv_metadata(join('data', "train.csv"), ltable=left, rtable=right, fk_ltable='ltable_id', fk_rtable='rtable_id')
train['id'] = range(0, len(train))
em.set_key(train, 'id')


#2. blocking
ob = em.OverlapBlocker()
K1 = ob.block_tables(left, right, 'title', 'title', l_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'], r_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'], overlap_size=3)
K2 = ob.block_candset(K1, 'brand', 'brand', overlap_size=1)


# 3. Feature Engineering
def jaccard_similarity(row, attr):
    x = set(row["ltable_" + attr].lower().split())
    y = set(row["rtable_" + attr].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row["ltable_" + attr].lower()
    y = row["rtable_" + attr].lower()
    return distance.nlevenshtein(x, y, method=2)


def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(K2)

left.index = left.id
right.index = right.id
pairs = np.array(list(map(tuple, train[["ltable_id", "rtable_id"]].values)))
tpls_l = left.loc[pairs[:, 0], :]
tpls_r = right.loc[pairs[:, 1], :]
tpls_l.columns = ["ltable_" + col for col in tpls_l.columns]
tpls_r.columns = ["rtable_" + col for col in tpls_r.columns]
tpls_l.reset_index(inplace=True, drop=True)
tpls_r.reset_index(inplace=True, drop=True)
training_df = pd.concat([tpls_l, tpls_r], axis=1)

training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=0)
rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)

# 5. output
matching_pairs = K2.loc[y_pred == 1, ["ltable_id", "rtable_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["ltable_id", "rtable_id"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("final.csv", index=False)
