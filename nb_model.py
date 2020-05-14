import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


data_variants = pd.read_csv('training/training_variants')
data_text = pd.read_csv("training/training_text", sep="\|\|", engine="python", names=["ID", "TEXT"], skiprows=1)

stop_words = set(stopwords.words('english'))

def report_log_loss(train_x, train_y, test_x, test_y, clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
    sig_clf.fit(train_x, train_y)
    sig_clf_probs = sig_clf.predict_proba(test_x)
    return log_loss(test_y, sig_clf_probs, eps=1e-15)

def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    
    A = (((C.T)/(C.sum(axis=1))).T)
    
    B = (C/C.sum(axis=0))
    labels = [1,2,3,4,5,6,7,8,9]
    print("-"*20, "Confusion Matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap='YlGnBu', fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    print("-"*20, "Precision Matrix(Column Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap='YlGnBu', fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    print("-"*20, "Recall Matrix(Row Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap='YlGnBu', fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

def get_impfeature_names(indices, text, gene, var, no_features):
    gene_count_vec = CountVectorizer()
    var_count_vec = CountVectorizer()
    text_count_vec = CountVectorizer(min_df = 3)
    
    gene_vec = gene_count_vec.fit(train_df('Gene'))
    var_vec = var_count_vec.fit(train_df('Variation'))
    text_vec = text_count_vec.fit(train_df('TEXT'))
    
    fea1_len = len(gene_vec.get_feature_names())
    fea2_len = len(var_count_vec.get_feature_names())
    
    word_present = 0
    for i, v in enumerate(indices):
        if (v < fea1_len):
            word = gene_vec.get_feature_names()[v]
            yes_no = True if word == gene else False
            if yes_no:
                word_present += 1
                print(i, "Gene feature [{}] present in test data point [{}]".format(word, yes_no))
        elif(v < fea1_len + fea2_len):
            word = var_vec.get_feature_names()[v-(fea1_len)]
            yes_no = True if word == var else False
            if yes_no:
                word_present += 1
                print(i, "Variation feature [{}] present in test data point [{}]".format(word, yes_no))
        else:
            word = text_vec.get_feature_names()[v-(fea1_len + fea2_len)]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
                print(i, "Text feature [{}] present in test data point [{}]".format(word, yes_no))
             
    print("Out of the top ",no_features, " features ", word_present, "are present in query point")

def data_text_preprocess(total_text, ind, col):
    if type(total_text) is not int:
        string = ""
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        total_text = re.sub('\s+', ' ', str(total_text))
        total_text = total_text.lower()
        
        for word in total_text.split():
            if not word in stop_words:
                string += word + " "
                
        data_text[col][ind] = string
        
for index, row in data_text.iterrows():
    if type(row['TEXT']) is str:
        data_text_preprocess(row['TEXT'], index, 'TEXT')
        
result = pd.merge(data_variants, data_text, on='ID', how='left')
        
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(), 'TEXT'] = result['Gene'] + ' ' + result['Variation']

y_true = result['Class'].values
result.Gene = result.Gene.str.replace('\s+', '_')
result.Variation = result.Gene.str.replace('\s+', '_')


gene_vectorizer = CountVectorizer()
result_gene = gene_vectorizer.fit_transform(result['Gene'])

variation_vectorizer = CountVectorizer()
result_variation = variation_vectorizer.fit_transform(result['Variation'])

text_vectorizer = CountVectorizer(min_df=3)
result_text = text_vectorizer.fit_transform(result['TEXT'])
result_text = normalize(result_text, axis=0)

result_gene = hstack((result_gene, result_variation))
result = hstack((result_gene, result_text)).tocsr()

X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2, random_state=43)
train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=43)

alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]
cv_log_error_array = []
for i in alpha:
    print("for alpha = ", i)
    clf = MultinomialNB(alpha=i)
    clf.fit(train_df, y_train)
    sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
    sig_clf.fit(train_df, y_train)
    sig_clf_probs=sig_clf.predict_proba(cv_df)
    cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))
    print("Log Loss : ", log_loss(y_cv, sig_clf_probs))
    
best_alpha = np.argmin(cv_log_error_array)
print("The best alpha : ", alpha[best_alpha])
clf = MultinomialNB(alpha=alpha[best_alpha])
clf.fit(train_df, y_train)
sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
sig_clf.fit(train_df, y_train)
sig_clf_probs=sig_clf.predict_proba(train_df)
print("For best alpha, the training log Loss : ", log_loss(y_train, sig_clf_probs))
sig_clf_probs=sig_clf.predict_proba(cv_df)
print("For best alpha, the CV log Loss : ", log_loss(y_cv, sig_clf_probs)) 
sig_clf_probs=sig_clf.predict_proba(test_df)
print("For best alpha, the testing log Loss : ", log_loss(y_test, sig_clf_probs)) 
 
sig_clf_probs=sig_clf.predict_proba(cv_df)
plot_confusion_matrix(y_cv, sig_clf.predict(cv_df.toarray()))

test_point_index = 1
no_features = 100
predicted_cls = sig_clf.predict(test_df[test_point_index])
print("Predicted Class: ", predicted_cls[0])
print("Predicted Class probabilities: ", np.round(sig_clf.predict_proba(test_df[test_point_index]), 4))
print("Actual Class: ", y_test[test_point_index])
indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_features]
print("-"*50)
get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index], test_df['Gene'].iloc[test_point_index], test_df['Variation'].iloc[test_point_index], no_features)
