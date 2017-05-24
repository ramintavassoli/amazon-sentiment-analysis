import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
import re
from pattern.en import tag
from sklearn.cluster import KMeans
from fractions import Fraction

#train
df = pd.read_csv('train_data.csv',
                header=0, sep=',',
                parse_dates=['date', 'release_date', 'update_date'],
                infer_datetime_format=True)
df = df[['review_text', 'star_rating']] 
#df.dropna(subset=['review_text'], inplace=True)
df = df.values.tolist()
stemmer = PorterStemmer()
for i in range(len(df)):
    temp = []
    for word, t in tag(df[i][0]):
        if t == 'JJ' or t == 'JJS' or t == 'JJR' or t == 'RB' or t == 'RBR' or t == 'RBS' or t == 'NN' or t == 'NNS':
            temp.append(word)
    df[i][0] = temp
    df[i][1] = int(df[i][1])

def score(lst):
    if len(lst) == 0:
        score = 0 
    else:
        score = 0
        for i in range(len(lst)):
            try:
                temp = (swn.senti_synset('{0}.a.01'.format(lst[i])).pos_score() + swn.senti_synset('{0}.n.01'.format(lst[i])).pos_score()) - (swn.senti_synset('{0}.a.01'.format(lst[i])).neg_score() + swn.senti_synset('{0}.n.01'.format(lst[i])).neg_score())
                score += temp
            except: 
                pass 
    return score

for i in range(len(df)):
        df[i][0] = score(df[i][0])

df = np.array(df)
temp = df[:,0]
new_df = [temp[i:i+1] for i in range(0, len(temp))]
kmeans = KMeans(n_clusters=5).fit(new_df)
sac_labels = kmeans.labels_
target = df[:,1]

sac0 = []
sac1 = []
sac2 = []
sac3 = []
sac4 = []

for i in range(len(sac_labels)):
    if sac_labels[i] == 0:
        sac0.append(target[i])
    elif sac_labels[i] == 1:
        sac1.append(target[i])
    elif sac_labels[i] == 2:
        sac2.append(target[i])
    elif sac_labels[i] == 3:
        sac3.append(target[i])
    elif sac_labels[i] == 4:
        sac4.append(target[i])

pmf01 = float(sac0.count(1.0))/len(sac0)
pmf02 = float(sac0.count(2.0))/len(sac0)
pmf03 = float(sac0.count(3.0))/len(sac0)
pmf04 = float(sac0.count(4.0))/len(sac0)
pmf05 = float(sac0.count(5.0))/len(sac0)
pmf0 = [pmf01,pmf02,pmf03,pmf04,pmf05]

pmf11 = float(sac1.count(1))/len(sac1)
pmf12 = float(sac1.count(2))/len(sac1)
pmf13 = float(sac1.count(3))/len(sac1)
pmf14 = float(sac1.count(4))/len(sac1)
pmf15 = float(sac1.count(5))/len(sac1)
pmf1 = [pmf11,pmf12,pmf13,pmf14,pmf15]

pmf21 = float(sac2.count(1))/len(sac2)
pmf22 = float(sac2.count(2))/len(sac2)
pmf23 = float(sac2.count(3))/len(sac2)
pmf24 = float(sac2.count(4))/len(sac2)
pmf25 = float(sac2.count(5))/len(sac2)
pmf2 = [pmf21,pmf22,pmf23,pmf24,pmf25]

pmf31 = float(sac3.count(1))/len(sac3)
pmf32 = float(sac3.count(2))/len(sac3)
pmf33 = float(sac3.count(3))/len(sac3)
pmf34 = float(sac3.count(4))/len(sac3)
pmf35 = float(sac3.count(5))/len(sac3)
pmf3 = [pmf31,pmf32,pmf33,pmf34,pmf35]

pmf41 = float(sac4.count(1))/len(sac4)
pmf42 = float(sac4.count(2))/len(sac4)
pmf43 = float(sac4.count(3))/len(sac4)
pmf44 = float(sac4.count(4))/len(sac4)
pmf45 = float(sac4.count(5))/len(sac4)
pmf4 = [pmf41,pmf42,pmf43,pmf44,pmf45]

#test
dft = pd.read_csv('test_data.csv',
                header=0, sep=',',
                parse_dates=['date', 'release_date', 'update_date'],
                infer_datetime_format=True)
dft = dft[['review_text']]
dft.dropna(subset=['review_text'], inplace=True)
dft = dft.values.tolist()
stemmer = PorterStemmer()
for i in range(len(dft)):
    temp = []
    for word, t in tag(dft[i][0]):
        if t == 'JJ' or t == 'JJS' or t == 'JJR' or t == 'RB' or t == 'RBR' or t == 'RBS' or t == 'NN' or t == 'NNS':
            temp.append(word)
    dft[i][0] = temp

for i in range(len(dft)):
        dft[i][0] = score(dft[i][0])

prediction = kmeans.predict(dft)    
prediction = prediction.tolist()
pred_lst = []
for i in range(len(prediction)):
    if prediction[i] == 0:
        pred_lst.append(pmf0)
    elif prediction[i] == 1:
        pred_lst.append(pmf1)
    elif prediction[i] == 2:
        pred_lst.append(pmf2)
    elif prediction == 3:
        pred_lst.append(pmf3)
    elif prediction[i] == 4:
        pred_lst.append(pmf4)

