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

# file = '../Model/df_medium_preprocessed.pkl'
file = '../../Project/df_020419_preprocessed_4.pkl'
df = pd.read_pickle(file)

df['class'] = np.nan
df['num_images'] = pd.to_numeric(df['num_images'])

bins = [[0,5],[5,10],[10,20],[20,50],[50,100],[100,200],[200,500],[500,1000],[1000,10000],[10000,1000000]]
for ibin,bin in enumerate(bins):
    print(bin)
    min = bin[0]
    max = bin[1]
    ind = np.where(np.logical_and(df.claps_num.values>=min, df.claps_num.values<max))
    for i in ind:
        df.loc[i,'class'] = ibin
    print(np.ma.count(df.claps_num.values[ind]),100*np.ma.count(df.claps_num.values[ind])/np.ma.count(df.claps_num.values),'%')

topic_class_modes = df.groupby('topic_num')['class'].apply(lambda x: x.mode()[0])
df['topic_class_mode'] = np.nan
for ind,topic in enumerate(df.topic_num):
  df.loc[ind,'topic_class_mode'] = topic_class_modes[topic]

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
X = X.merge(sentiment_pos, on='index')
X = X.merge(sentiment_neg, on='index')
X = X.merge(sentiment_neu, on='index')
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

X_train, X_test, y_train, y_test, y_topic_mode_train, y_topic_mode_test = train_test_split(
                                    X, y, y_topic_mode, test_size=0.2, random_state=4)
                                    
print('Train X and y: ', X_train.shape, y_train.shape)
print('Test X and y: ', X_test.shape, y_test.shape)

sm = SMOTENC(categorical_features=[1], sampling_strategy='auto',
             random_state=4, k_neighbors=5, n_jobs=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
X_train_res = pd.DataFrame(X_train_res,columns = X.columns)
print('Resampled train X and y: ', X_train_res.shape, y_train_res.shape)

param_grid = {}
param_grid['application'] = 'multiclass'
param_grid['num_class'] = len(bins)
param_grid['learning_rate'] = 0.01
param_grid['boosting_type'] = 'gbdt'
param_grid['metric'] = 'multiclass'
param_grid['lambda_l2'] = 50.0
param_grid['sub_feature'] = 0.8
param_grid['bagging_fraction'] = 0.8
param_grid['num_leaves'] = 40
param_grid['max_depth'] = 10
param_grid['random_state'] = 3

X_train_res.loc[:,'topic_num'] = X_train_res['topic_num'].astype('category').copy()
X_test.loc[:,'topic_num'] = X_test['topic_num'].astype('category').copy()

train_data = lgbm.Dataset(X_train_res,y_train_res)
test_data = lgbm.Dataset(X_test, y_test, reference = train_data)
# train_data = lgbm.Dataset(X_train_res,y_train_res, categorical_feature=['topic_num'])
# test_data = lgbm.Dataset(X_test, y_test, reference = train_data, categorical_feature=['topic_num'])

model_lgbm = lgbm.train(param_grid, train_data, num_boost_round=2000, 
                        valid_sets=[train_data, test_data], early_stopping_rounds=20, 
                        verbose_eval=50, categorical_feature=['topic_num'])

model_lgbm.save_model('../Model/lgbm_model')

y_pred = np.argmax(model_lgbm.predict(X_test),axis=1)
X_train.loc[:,'topic_num'] = X_train['topic_num'].astype('category').copy()
y_train_pred = np.argmax(model_lgbm.predict(X_train),axis=1)

prec = metrics.precision_score(y_test,y_pred, average='weighted')
prec_train = metrics.precision_score(y_train, y_train_pred, average='weighted')
prec_mean = metrics.precision_score(y_test, y_topic_mode_test, average='weighted')
print('\n')
print('Precision score on training data: ', prec_train)
print('Precision score on test data: ', prec)
print('Precision score on test data, assuming mode for topic: ', prec_mean)
print('Model does '+"{:.2f}".format(100*(prec-prec_mean)/prec_mean)+' % better than guessing most common class for each topic')

bal = metrics.balanced_accuracy_score(y_test,y_pred)
bal_train = metrics.balanced_accuracy_score(y_train, y_train_pred)
bal_mean = metrics.balanced_accuracy_score(y_test, y_topic_mode_test)
print('\n')
print('Balanced accuracy score on training data: ', bal_train)
print('Balanced accuracy score on test data: ', bal)
print('Balanced accuracy score on test data, assuming mode for topic: ', bal_mean)
print('Model does '+"{:.2f}".format(100*(bal-bal_mean)/bal_mean)+' % better than guessing most common class for each topic')

recall = metrics.recall_score(y_test,y_pred, average='weighted')
recall_train = metrics.recall_score(y_train, y_train_pred, average='weighted')
recall_mean = metrics.recall_score(y_test, y_topic_mode_test, average='weighted')
print('\n')
print('Recall score on training data: ', recall_train)
print('Recall score on test data: ', recall)
print('Recall score on test data, assuming mode for topic: ', recall_mean)
print('Model does '+"{:.2f}".format(100*(recall-recall_mean)/recall_mean)+' % better than guessing most common class for each topic')

f1 = metrics.f1_score(y_test,y_pred, average='weighted')
f1_train = metrics.f1_score(y_train, y_train_pred, average='weighted')
f1_mean = metrics.f1_score(y_test, y_topic_mode_test, average='weighted')
print('\n')
print('f1 score on training data: ', f1_train)
print('f1 score on test data: ', f1)
print('f1 score on test data, assuming mode for topic: ', f1_mean)
print('Model does '+"{:.2f}".format(100*(f1-f1_mean)/f1_mean)+' % better than guessing most common class for each topic')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
fig,ax = plt.subplots(figsize=(7,5))
cm_train = metrics.confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm_train, annot=True)
plt.savefig('../Figs/train_confusion.png')

fig,ax = plt.subplots(figsize=(7,5))
cm_test = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm_test, annot=True)
plt.savefig('../Figs/test_confusion.png')

class_names = ['0-5','5-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-10000','>10000']
plt.figure()
plot_confusion_matrix(cm_train, classes=class_names, normalize=True,
                      title='Normalized confusion matrix - train')
plt.savefig('../Figs/train_confusion_normalized.png')

plt.figure()
plot_confusion_matrix(cm_test, classes=class_names, normalize=True,
                      title='Normalized confusion matrix - test')
plt.savefig('../Figs/test_confusion2_normalized.png')
