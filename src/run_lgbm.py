# This script reads a dataframe of pre-processed Medium articles
# Number of claps are bucketed into 10 bins
# SMOTE upsampling is performed on the minority classes to balance the classes
# Train gradient-boosted decision tree using LightGBM
# Calculate model performance metrics

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import imblearn
from imblearn.over_sampling import SMOTENC
import seaborn as sns
import matplotlib.pyplot as plt
import os
import itertools

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Read in preprocessed Medium articles

file = '../Model/df_medium_preprocessed.pkl'
df = pd.read_pickle(file)

df = df[df.word_count > 300].copy()
df.reset_index(inplace=True)
df['num_images'] = pd.to_numeric(df['num_images'])

# Bucket clap counts into 10 bins

bins = [[0,5],[5,10],[10,20],[20,50],[50,100],[100,200],[200,500],[500,1000],[1000,10000],[10000,1000000]]
for ibin,bin in enumerate(bins):
    print(bin)
    min = bin[0]
    max = bin[1]
    ind = np.where(np.logical_and(df.claps_num.values>=min, df.claps_num.values<max))
    for i in ind:
        df.loc[i,'class'] = ibin
    print(np.ma.count(df.claps_num.values[ind]),100*np.ma.count(df.claps_num.values[ind])/np.ma.count(df.claps_num.values),'%')

# Calculate most common bin for each topic

topic_class_modes = df.groupby('topic_num')['class'].apply(lambda x: x.mode()[0])
df['topic_class_mode'] = np.nan
for ind,topic in enumerate(df.topic_num):
  df.loc[ind,'topic_class_mode'] = topic_class_modes[topic]
  
# Calculate feature matrix X and target vector y

topic_cat = pd.DataFrame(df.topic_num).reset_index()
sentiment_compound = df['compound'].reset_index()
sentiment_pos = df['pos'].reset_index()
sentiment_neg = df['neg'].reset_index()
sentiment_neu = df['neu'].reset_index()
reading_time = df['reading_time_num'].reset_index()
readability_index = df.readability_index.reset_index()
word_count = df.word_count.reset_index()
unique_word_count = df.unique_word_count.reset_index()
average_word_length = df.average_word_length.reset_index()
sentence_count = df.sentence_count.reset_index()
average_sentence_length = df.average_sentence_length.reset_index()
facebook_shares = df.facebook_shares.reset_index()
num_images = df.num_images.reset_index()

X = topic_cat.merge(sentiment_compound, on='index')
X = X.merge(readability_index, on='index')
X = X.merge(word_count, on='index')
X = X.merge(unique_word_count, on='index')
X = X.merge(average_word_length, on='index')
X = X.merge(sentence_count, on='index')
X = X.merge(average_sentence_length, on='index')
X = X.merge(facebook_shares, on='index')
X = X.merge(num_images, on='index')

y = df['class']
X = X.drop(['index'],axis=1)

y_topic_mode = df['topic_class_mode']
  
print('X and y shapes: ', X.shape, y.shape)

# Split into training and testing data with an 80/20 split

X_train, X_test, y_train, y_test, y_topic_mode_train, y_topic_mode_test = train_test_split(
                                    X, y, y_topic_mode, test_size=0.2, random_state=4)
                                    
print('Train X and y: ', X_train.shape, y_train.shape)
print('Test X and y: ', X_test.shape, y_test.shape)

# Upsample minority classes using SMOTE

sm = SMOTENC(categorical_features=[0], sampling_strategy='auto',
             random_state=4, k_neighbors=5, n_jobs=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
X_train_res = pd.DataFrame(X_train_res,columns = X.columns)

print('Resampled train X and y: ', X_train_res.shape, y_train_res.shape)

# Set parameters for gradient boosting 

param_grid = {}
param_grid['application'] = 'multiclass'
param_grid['num_class'] = len(bins)
param_grid['learning_rate'] = 0.01
param_grid['boosting_type'] = 'gbdt'
param_grid['metric'] = 'multi_logloss'
param_grid['lambda_l2'] = 50.0
param_grid['sub_feature'] = 0.8
param_grid['bagging_fraction'] = 0.8
param_grid['num_leaves'] = 40
param_grid['max_depth'] = 10
param_grid['random_state'] = 3

X_train_res.loc[:,'topic_num'] = X_train_res['topic_num'].astype('category').copy()
X_test.loc[:,'topic_num'] = X_test['topic_num'].astype('category').copy()

# Calculate datasets for gradient boosting 

train_data = lgbm.Dataset(X_train_res,y_train_res)
test_data = lgbm.Dataset(X_test, y_test, reference = train_data)

# Perform gradient boosting 

model_lgbm = lgbm.train(param_grid, train_data, num_boost_round=2000, 
                        valid_sets=[train_data, test_data], early_stopping_rounds=20, 
                        verbose_eval=50, categorical_feature=['topic_num'])

model_lgbm.save_model('../Model/lgbm_model')

# Predict targets using models 
# (Predicted class is the one with the highest probability)

y_pred = np.argmax(model_lgbm.predict(X_test),axis=1)
y_pred0 = model_lgbm.predict(X_test)
X_train.loc[:,'topic_num'] = X_train['topic_num'].astype('category').copy()
y_train_pred = np.argmax(model_lgbm.predict(X_train),axis=1)
y_train_pred0 = model_lgbm.predict(X_train)

# Calculate performance metrics

# Mean reciprocal rank

y_test2 = y_test.copy().reset_index()
mrrsum = 0
for i in range(len(y_test2)):
  y_true = y_test2['class'][i]
  idx = np.argsort(y_pred0[i])[::-1]
  class_order = np.array(range(len(bins)))[idx]
  position = list(class_order).index(int(y_true))
  mrrsum += 1.0/(position+1.0)
mrr = mrrsum/len(y_pred)

y_train2 = y_train.copy().reset_index()
mrrsum = 0
for i in range(len(y_train2)):
  y_true = y_train2['class'][i]
  idx = np.argsort(y_train_pred0[i])[::-1]
  class_order = np.array(range(len(bins)))[idx]
  position = list(class_order).index(int(y_true))
  mrrsum += 1.0/(position+1.0)
mrr_train = mrrsum/len(y_train2)

print('MRR on training data: ', mrr_train)
print('MRR on test data: ', mrr)

# Modified mean reciprocal rank

y_topic_mode_test2 = y_topic_mode_test.copy().reset_index()
mrrsum = 0
mrrsum_topic = 0
mrrsum_0 = 0
for i in range(len(y_test2)):
  y_true = y_test2['class'][i]
  idx = np.argsort(y_pred0[i])[::-1]
  class_order = np.array(range(len(bins)))[idx]
  position = list(class_order).index(int(y_true))
  mrrsum += 1.0/(abs(y_true-y_pred[i])+1.0)
  mrrsum_topic += 1.0/(abs(y_true-y_topic_mode_test2['topic_class_mode'][i])+1.0)
  mrrsum_0 += 1.0/(abs(y_true-0)+1.0)
mrr2 = mrrsum/len(y_test2)
mrr2_topic = mrrsum_topic/len(y_test2)
mrr2_0 = mrrsum_0/len(y_test2)

mrrsum = 0
for i in range(len(y_train2)):
  y_true = y_train2['class'][i]
  idx = np.argsort(y_train_pred0[i])[::-1]
  class_order = np.array(range(len(bins)))[idx]
  position = list(class_order).index(int(y_true))
  mrrsum += 1.0/(abs(y_true-y_train_pred[i])+1.0)
mrr2_train = mrrsum/len(y_train2)

print('MRR2 on training data: ', mrr2_train)
print('MRR2 on test data: ', mrr2)
print('MRR2 on test data, assuming mode for topic: ', mrr2_topic)
print('Model does '+"{:.2f}".format(100*(mrr2-mrr2_topic)/mrr2_topic)+' % better than guessing most common class for each topic')
print('MRR2 on test data, assuming 0 for topic class: ', mrr2_0)

# Calculate "fuzzy" predicted class, where it is considered a success 
# if the prediction is off only by one class

y_zeros = np.zeros(len(y_pred))
y_pred_fuzzy = y_pred.copy()
y_pred_topic_fuzzy = y_topic_mode_test2['topic_class_mode'].copy()
for idx,pred in enumerate(y_pred_fuzzy):
  if abs(pred-y_test2['class'][idx]) <= 1:
    y_pred_fuzzy[idx] = y_test2['class'][idx]
  if abs(y_pred_topic_fuzzy[idx]-y_test2['class'][idx]) <= 1:
    y_pred_topic_fuzzy[idx] = y_test2['class'][idx]

# Precision

prec = metrics.precision_score(y_test,y_pred, average='weighted')
prec_fuzzy = metrics.precision_score(y_test,y_pred_fuzzy, average='weighted')
prec_train = metrics.precision_score(y_train, y_train_pred, average='weighted')
prec_mean = metrics.precision_score(y_test, y_topic_mode_test, average='weighted')
prec_mean_fuzzy = metrics.precision_score(y_test, y_pred_topic_fuzzy, average='weighted')
print('\n')
print('Precision score on training data: ', prec_train)
print('Precision score on test data: ', prec)
print('Precision score on test data, assuming mode for topic: ', prec_mean)
print('Model does '+"{:.2f}".format(100*(prec-prec_mean)/prec_mean)+' % better than guessing most common class for each topic')
print('Precision score on test data, if its ok to be off by 1 class: ', prec_fuzzy)
print('Precision score on test data, assuming mode for topic, if its ok to be off by 1 class: ', prec_mean_fuzzy)
print('Model does '+"{:.2f}".format(100*(prec_fuzzy-prec_mean_fuzzy)/prec_mean_fuzzy)+' % better than guessing most common class for each topic')

# Balanced accuracy score

bal = metrics.balanced_accuracy_score(y_test,y_pred)
bal_fuzzy = metrics.balanced_accuracy_score(y_test,y_pred_fuzzy)
bal_train = metrics.balanced_accuracy_score(y_train, y_train_pred)
bal_mean = metrics.balanced_accuracy_score(y_test, y_topic_mode_test)
bal_mean_fuzzy = metrics.balanced_accuracy_score(y_test, y_pred_topic_fuzzy)
print('\n')
print('Balanced accuracy score on training data: ', bal_train)
print('Balanced accuracy score on test data: ', bal)
print('Balanced accuracy score on test data, assuming mode for topic: ', bal_mean)
print('Model does '+"{:.2f}".format(100*(bal-bal_mean)/bal_mean)+' % better than guessing most common class for each topic')
print('Balanced accuracy score on test data, if its ok to be off by 1 class: ', bal_fuzzy)
print('Balanced accuracy score on test data, assuming mode for topic, if its ok to be off by 1 class: ', bal_mean_fuzzy)
print('Model does '+"{:.2f}".format(100*(bal_fuzzy-bal_mean_fuzzy)/bal_mean_fuzzy)+' % better than guessing most common class for each topic')

# Recall

recall = metrics.recall_score(y_test,y_pred, average='weighted')
recall_fuzzy = metrics.recall_score(y_test,y_pred_fuzzy, average='weighted')
recall_train = metrics.recall_score(y_train, y_train_pred, average='weighted')
recall_mean = metrics.recall_score(y_test, y_topic_mode_test, average='weighted')
recall_mean_fuzzy = metrics.recall_score(y_test, y_pred_topic_fuzzy, average='weighted')
print('\n')
print('Recall score on training data: ', recall_train)
print('Recall score on test data: ', recall)
print('Recall score on test data, assuming mode for topic: ', recall_mean)
print('Model does '+"{:.2f}".format(100*(recall-recall_mean)/recall_mean)+' % better than guessing most common class for each topic')
print('Recall score on test data, if its ok to be off by 1 class: ', recall_fuzzy)
print('Recall score on test data, assuming mode for topic, if its ok to be off by 1 class: ', recall_mean_fuzzy)
print('Model does '+"{:.2f}".format(100*(recall_fuzzy-recall_mean_fuzzy)/recall_mean_fuzzy)+' % better than guessing most common class for each topic')

# f1 score

f1 = metrics.f1_score(y_test,y_pred, average='weighted')
f1_fuzzy = metrics.f1_score(y_test,y_pred_fuzzy, average='weighted')
f1_train = metrics.f1_score(y_train, y_train_pred, average='weighted')
f1_mean = metrics.f1_score(y_test, y_topic_mode_test, average='weighted')
f1_mean_fuzzy = metrics.f1_score(y_test, y_pred_topic_fuzzy, average='weighted')
f1_zero = metrics.f1_score(y_test, y_zeros, average='weighted')
print('\n')
print('f1 score on training data: ', f1_train)
print('f1 score on test data: ', f1)
print('f1 score on test data, assuming mode for topic: ', f1_mean)
print('Model does '+"{:.2f}".format(100*(f1-f1_mean)/f1_mean)+' % better than guessing most common class for each topic')
print('f1 score on test data, assuming 0 for topic class: ', f1_zero)
print('f1 score on test data, if its ok to be off by 1 class: ', f1_fuzzy)
print('f1 score on test data, assuming mode for topic, if its ok to be off by 1 class: ', f1_mean_fuzzy)
print('Model does '+"{:.2f}".format(100*(f1_fuzzy-f1_mean_fuzzy)/f1_mean_fuzzy)+' % better than guessing most common class for each topic')
  
