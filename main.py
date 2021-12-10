import json
import math
import re
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
import collections
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

#Load the data
f = open("TVs-all-merged.json")
data = json.load(f)

#Define function that loops over the dictionary and splits the modelID (keys) from the rest (values)
def pairs_dict(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            for product in pairs_dict(value):
                yield (key, *product)
        else:
            yield (key, value)

#Loop over every product in the dictionary and extract the title
counts = {}
titles = []
validation_df = pd.DataFrame(columns=['model ID', 'title'])
for pair in pairs_dict(data):
    for item_number in range(len(pair[1])):
        title = pair[1][item_number]['title']

        #Remove all modelIDs from titles
        model_id = pair[0]
        if model_id in title:
            title = title.replace(model_id, '')

        #Remove and replace values for clarity
        title = re.sub('(/ |- |Newegg.com| TheNerds.net|Best Buy|,|\+|:|\)|\(| tv| TV|Class |diag\. |Diag. |diagonal )', '', title)
        title = re.sub('(-inch|inch|"|Inch|inches| inch| Inch|-Inch)', 'inch', title)
        title = re.sub('(Hz|hz|Hertz|hertz|HZ| hz|-hz| Hz)', 'hz', title)
        title = re.sub('(ledlcd |LEDLCD|LED-LCD)', 'LED LCD', title)

        #Count the number of words in the titles
        for word in title.split():
            word = word.lower()
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        titles.append(title.lower())

        validation_df = validation_df.append({'model ID':model_id, 'title':title}, ignore_index=True)

duplicate_pairs = 0
valuecount = validation_df['model ID'].value_counts()
for modelcount in valuecount:
    if modelcount == 2:
        duplicate_pairs += 1
    elif modelcount == 3:
        duplicate_pairs += 3
    elif modelcount == 4:
        duplicate_pairs += 6
    else:
        continue
print(duplicate_pairs)

#Sort counts and create a new dictionary with words that only appear once
wordcounts_one = {k:v for k,v in counts.items() if v == 1}
wordcounts = {k:v for k,v in counts.items() if v != 1}

#Remove words that only appear once
titles_final = []
for title in titles:
    for word1 in wordcounts_one.keys():
        if word1 in title.split():
            title = title.replace(word1, '')
    titles_final.append(title)

#Remove spaces at the beginning and end of the title
count = 0
for title in titles_final:
    if title.startswith(' '):
        title = title[1:]
    if title.startswith(' '):
        title = title[1:]
    if title.endswith(' '):
        title = title[:-1]
    if title.endswith(' '):
        title = title[:-1]
    titles_final[count] = title
    count += 1

#Split titles on space and put each word within a title in a separate list
titles_split = []
for title in titles_final:
    titles_split.append(title.split())

#Make matrix of binary vector representations of titles
mlb = MultiLabelBinarizer()
df = pd.DataFrame({"titles": titles_split})
df = pd.DataFrame(mlb.fit_transform(df['titles']), columns=mlb.classes_, index=df.index)
df = df.transpose()
deleted_titles = df.loc[:, (df == 0).all(axis=0)]
df = df.loc[:, (df != 0).any(axis=0)]
df = df.transpose()
df.reset_index(drop=True, inplace=True)
df = df.transpose()
for title in deleted_titles:
    validation_df = validation_df[validation_df.index != title]
validation_df.reset_index(drop=True, inplace=True)

signature_matrix = np.full((len(df.columns), 150), np.inf)
hash_functions = []
#Set seed
np.random.seed(20)
for row in range(len(df)):
    hash_row = []
    for i in range(150):
        np.random.seed(1)
        int1 = random.randint(0, 100)
        np.random.seed(2)
        int2 = random.randint(0, 100)
        hash_value = (int1 + int2 * (row + 1)) % 887
        hash_row.append(hash_value)
    hash_functions.append(hash_row)
    for column in df.columns:
        if (df.iloc[row][column] == 1):
            for i in range(len(hash_functions[row])):
                value = hash_functions[row][i]
                if value < signature_matrix[column][i]:
                    signature_matrix[column][i] = value
signmatrix = signature_matrix.transpose()

def intersection_list(lst1, lst2):
    list_intersec = [value for value in range(len(lst1)) if lst2[value] == 1 and lst1[value] == 1]
    return list_intersec

def union_list(lst1, lst2):
    try:
        list_union = lst1.value_counts()[1] + lst2.value_counts()[1] - len(intersection_list(lst1,lst2))
    except:
        print(lst1.value_counts())
        print(lst2.value_counts())
        print(lst1.value_counts()[1])
        print(lst2.value_counts()[1])
    return list_union

def jaccardSim(d1,d2):
    return len(intersection_list(d1,d2))/union_list(d1,d2)

b=30
r=5
n, d = signmatrix.shape
assert(n==b*r)
hashbuckets = collections.defaultdict(set)
bands = np.array_split(signmatrix, b, axis=0)
for k,band in enumerate(bands):
    for j in range(d):
        band_id = tuple(list(band[:,j])+[str(k)])
        hashbuckets[band_id].add(j)
candidate_pairs = set()
for bucket in hashbuckets.values():
    if len(bucket) > 1:
        for pair in itertools.combinations(bucket, 2):
            candidate_pairs.add(pair)

lsh_pairs = set()
threshold = (1/b)**(1/r)
for tuple in candidate_pairs:
    if jaccardSim(df[tuple[0]], df[tuple[1]]) > threshold:
        lsh_pairs.add((tuple[0], tuple[1]))
print(lsh_pairs)
print(len(lsh_pairs))

classification_df = pd.DataFrame(columns=['candidates', 'duplicate label'])
for lshpair in lsh_pairs:
    array1 = df[lshpair[0]].to_list()
    array2 = df[lshpair[1]].to_list()
    array = array1 + array2
    if validation_df['model ID'][lshpair[0]] == validation_df['model ID'][lshpair[1]]:
        classification_df = classification_df.append({'candidates': array,'duplicate label': 1}, ignore_index=True)
    else:
        classification_df = classification_df.append({'candidates': array,'duplicate label': 0}, ignore_index=True)
classification_df.to_csv('class_df.csv')
print(classification_df['duplicate label'].value_counts())

X = pd.DataFrame(columns=range(670))
for index, lst in enumerate(classification_df.candidates):
        X.loc[index] = lst
X.to_csv('X.csv')
Y = pd.Series(classification_df['duplicate label'].astype('int'))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.37)
base_estimator = ExtraTreesClassifier(random_state=42)
param_grid = {'max_features': ['sqrt', 'None'],
              'n_estimators': [100, 200, 500],
              'criterion': ['gini', 'entropy']
              }
GS = GridSearchCV(base_estimator, param_grid, cv=5, scoring='accuracy')
GS.fit(X_train, Y_train)
print(f'Best parameters: {GS.best_params_}')

f1score = []
for bootstrap in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.37)
    ET = ExtraTreesClassifier(random_state=42, n_estimators=200,
                              criterion='gini').fit(X_train, Y_train)
    y_pred =ET.predict(X_test)
    score = f1_score(Y_test, y_pred)
    f1score.append(score)
finalprediction = np.mean(f1score)
print(finalprediction)