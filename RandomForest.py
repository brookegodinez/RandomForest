#Code and ideas adapted from "https://carbonati.github.io/posts/random-forests-from-scratch/"" 

import json
import random
import spacy

from models import repo
nlp = spacy.load("en_core_web_sm")
from bag_of_word import json_to_model, get_problem_explanation, load_json, create_bow_matrix, bow_column_list
from random import randrange
import numpy as np
import math as math 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class RandomForest: 
    def __init__(self, X_train, y_train, n_estimators, max_features, max_depth, min_split):
        self.X_train = X_train
        self.y_train = y_train
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_split = min_split


    def random_forest(self):
        tree_list = []
        OOB_list = []
        for i in range(self.n_estimators):
            X_bootstrap, y_bootstrap, X_oob, y_oob = self.bootstrap(self.X_train, self.y_train)

            tree = self.decision_tree(X_bootstrap, y_bootstrap, self.max_features, self.max_depth, self.min_split)

            tree_list.append(tree)
            oobError = self.OOB(tree, X_oob, y_oob)
            OOB_list.append(oobError)

        print("Out of bag error estimate: {:.3f}".format(np.mean(OOB_list)))
        return tree_list

    def entropy(self, p):
        if p == 0 or p == 1:
            return 0
        else:
            return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

    def info_gain(self, left_c, right_c):
        parent = left_c + right_c
        # print(type(parent))
        if len(parent) > 0:
            p_parent = parent.count(1) / len(parent) 
        else:
            p_parent =  0
        if len(left_c) > 0:
            p_left = left_c.count(1) / len(left_c)
        else:
            p_left = 0 
        if len(right_c) > 0:
            p_right = right_c.count(1) / len(right_c)
        else:
            p_right = 0 
        info_p = self.entropy(p_parent)
        info_l = self.entropy(p_left)
        info_r = self.entropy(p_right)
        return info_p - len(left_c) / len(parent) * info_l - len(right_c) / len(parent) * info_r


    def bootstrap(self, X_train, y_train):
        OOB_indices = []   
        bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
        for i in range(len(X_train)):
            if i not in bootstrap_indices:
                OOB_indices.append(i)
        X_bootstrap = X_train.iloc[bootstrap_indices].values
        # print(X_bootstrap)
        # print("+++++++++++++++++++++++++++")
        X_oob = X_train.iloc[OOB_indices].values
        Y_bootstrap = y_train.iloc[bootstrap_indices].values
        y_oob = y_train.iloc[OOB_indices].values
        return X_bootstrap, Y_bootstrap, X_oob, y_oob

    def OOB(self, tree, X_test, Y_test):
        mis_label = 0
        # print(X_test)
        for i in range(len(X_test)):
            pred = self.predict_tree(tree, X_test[i])
            if pred != Y_test[i]:
                mis_label += 1
        return mis_label / len(X_test)

    def find_split(self, X_bootstrap,Y_bootstrap, max):
        best_gain = -9999
        feature_list = []
        # print(X_bootstrap[0])
        num_features = len(X_bootstrap[0])
        while len(feature_list) <= max:
            feature_index = random.sample(range(num_features), 1)
            if feature_index not in feature_list:
                feature_list.extend(feature_index)
        node = None
        for feature_idx in feature_list:
            for split_point in X_bootstrap[:,feature_idx]:
                # print("split point:", split_point) #
                left = {'X_bootstrap': [], 'y_bootstrap': []}
                right = {'X_bootstrap': [], 'y_bootstrap': []}

                if type(split_point) in [int, float]:
                    for i, value in enumerate(X_bootstrap[:,feature_idx]):
                        if value <= split_point:
                            left['X_bootstrap'].append(X_bootstrap[i])
                            left['y_bootstrap'].append(Y_bootstrap[i])
                        else:
                            right['X_bootstrap'].append(X_bootstrap[i])
                            right['y_bootstrap'].append(Y_bootstrap[i])
                
                else:
                    for i, value in enumerate(X_bootstrap[:,feature_idx]):
                        if value == split_point:
                            left['X_bootstrap'].append(X_bootstrap[i])
                            left['y_bootstrap'].append(Y_bootstrap[i])
                        else:
                            right['X_bootstrap'].append(X_bootstrap[i])
                            right['y_bootstrap'].append(Y_bootstrap[i])

                split_gain = self.info_gain(left['y_bootstrap'], right['y_bootstrap'])
                if split_gain > best_gain:
                    best_gain = split_gain
                    left['X_bootstrap'] = np.array(left['X_bootstrap'], dtype=object)
                    right['X_bootstrap'] = np.array(right['X_bootstrap'], dtype=object)
                    node = {'information_gain': split_gain, 'left_child': left, 'right_child': right,
                    'split_point': split_point, 'feature_idx': feature_idx}

        return node
        
    def terminal_node(self, node):
        # print("in terminal node")
        y_bootstrap = node['y_bootstrap']
        prediction = max(y_bootstrap, key = y_bootstrap.count)
        return prediction


    def node_split(self, node, max_features, min_samples_split, max_depth, depth):
        # print("in split node")
        left_child = node['left_child']
        right_child = node['right_child']

        del(node['left_child'])
        del(node['right_child'])

        if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
            empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
            node['left_split'] = self.terminal_node(empty_child)
            node['right_split'] = self.terminal_node(empty_child)
            return

        if depth >= max_depth:
            node['left_split'] = self.terminal_node(left_child)
            node['right_split'] = self.terminal_node(right_child)
            return node

        if len(left_child['X_bootstrap']) <= min_samples_split:
            node['left_split'] = node['right_split'] = self.terminal_node(left_child)
        else:
            node['left_split'] = self.find_split(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
            self.node_split(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
        if len(right_child['X_bootstrap']) <= min_samples_split:
            node['right_split'] = node['left_split'] = self.terminal_node(right_child)
        else:
            node['right_split'] = self.find_split(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
            self.node_split(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)
        
    def decision_tree(self, X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    
        root = self.find_split(X_bootstrap, y_bootstrap, max_features)
        # print(root_node)
        self.node_split(root, max_features, min_samples_split, max_depth, 1)
        return root


    def predict_tree(self, tree, X_test):
        # print(X_test)
        feature_index = tree['feature_idx']
        # print("feature_idx:", feature_idx)
        # print(tree['split_point'])
        if (feature_index >= len(X_test)):
            feature_index = len(X_test) - 1
        # print(X_test[feature_idx])
        if X_test[feature_index] <= tree['split_point']:
        
            if type(tree['left_split']) == dict:
                return self.predict_tree(tree['left_split'], X_test)
            else:
                value = tree['left_split']
                return value
        else:
        
            if type(tree['right_split']) == dict:
                return self.predict_tree(tree['right_split'], X_test)
            else:
                return tree['right_split']

    def predict(self, tree_list, X_test):
        pred_list = list()
        for i in range(len(X_test)):
            ensemble_predicts = [self.predict_tree(tree, X_test.values[i]) for tree in tree_list]
            final_predict = max(ensemble_predicts, key = ensemble_predicts.count)
            pred_list.append(final_predict)
        return np.array(pred_list)


def train_test_split(data, test_size):
    train = list()
    train_size = test_size * len(data)
    copy_of_data = list(data)
    while len(train) < train_size:
        index = randrange(len(copy_of_data))
        # if copy_of_data:
        train.append(copy_of_data.pop(index))
    return train , copy_of_data


n_estimators = 100

# max_features = 3
max_depth = 10
min_samples_split = 2
data = load_json("results.json")

repo_list = json_to_model(data)




training_data, test_data  = train_test_split(repo_list, .80)

# category_table = order_filter(training_data)
category_table = get_problem_explanation(training_data)
category = bow_column_list(category_table)

####Creates the matrix of our data, with each feature representing a signifigant word
####in the explination. The values respresent how many time each word is within the 
### explination for a particular category. 
####howeever gives a very sparse matrix
matrix = create_bow_matrix(category_table, category) 

## a list of all features of our matrix to use in doing PCA on the data
list_of_features = []
for i in matrix:
    if i != 'labels':
        list_of_features.append(i)
    
#a matrix contain just our values sans category and collumn 
value = matrix.loc[:,list_of_features].values

####The values for our data are scaled 
####MINMAXSCALAR fit_transform scales each feature to a given range
####
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(value)

### I used the way the scaler is fit to determine how many components I need for my data
##explained variance should be bewteen 95-99% 
pca_matrix = PCA(n_components= .95)
pca_matrix = pca_matrix.fit_transform(data_rescaled)

pca_matrix_df = pd.DataFrame(data = pca_matrix)
# print(pca_matrix_df)


#####These steps are necessary as our original matrix is so spares it takes a long time for even
###our small data set to run through the tree and helps us to avoid the other problems that come with too many features
###Like overfitting 

X_train = pca_matrix_df
Y_train = matrix['labels']

max_features = 3 
# print(int(max_features))
model = RandomForest(X_train, Y_train, n_estimators, int(max_features), max_depth, min_samples_split)
model.random_forest()
