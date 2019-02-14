import numpy as np
import pandas as pd
import lightgbm as lgbm
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def calculate_prediction_change(model, df, bins, feature, perturb):
  df2 = df.copy()
  df2.loc[0,feature] = df2[feature][0] + perturb
  if df2.loc[0,feature] >= 0:
    predicted_claps_class = int(np.argmax(model.predict(df2),axis=1))
    max_claps = bins[predicted_claps_class][1]
  else:
    max_claps = 0
  return max_claps

def find_min_perturb(model, df, bins, feature, minperturb, max_claps):
    iter = 0
    max_claps2 = 0
    while max_claps2 <= max_claps:
        iter += 1
        perturb = minperturb * iter
        max_claps2 = calculate_prediction_change(model, df, bins, feature, perturb)
        print(iter,perturb,max_claps,max_claps2)
        if max_claps2 < max_claps or iter > 100:
            max_claps2 = 0
            break
    return perturb,max_claps2    

def calculate_model_prediction(df):
  
  topic_cat = pd.DataFrame(df.topic_num).reset_index()
  sentiment_compound = df['compound'].reset_index()
  sentiment_pos = df['pos'].reset_index()
  sentiment_neg = df['neg'].reset_index()
  sentiment_neu = df['neu'].reset_index()
  readability_index = df.readability_index.reset_index()
  word_count = df.word_count.reset_index()
  unique_word_count = df.unique_word_count.reset_index()
  average_word_length = df.average_word_length.reset_index()
  sentence_count = df.sentence_count.reset_index()
  average_sentence_length = df.average_sentence_length.reset_index()
  facebook_shares = df.facebook_shares.reset_index()
  num_images = df.num_images.reset_index()

  X = topic_cat.merge(sentiment_compound, on='index')
#   X = X.merge(sentiment_pos, on='index')
#   X = X.merge(sentiment_neg, on='index')
#   X = X.merge(sentiment_neu, on='index')
  X = X.merge(readability_index, on='index')
  X = X.merge(word_count, on='index')
  X = X.merge(unique_word_count, on='index')
  X = X.merge(average_word_length, on='index')
  X = X.merge(sentence_count, on='index')
  X = X.merge(average_sentence_length, on='index')
  X = X.merge(facebook_shares, on='index')
  X = X.merge(num_images, on='index')
  X = X.drop(['index'],axis=1)
  
  print('Types00: ')
  print(X.dtypes)
  print(X['topic_num'])
#   X.loc[:,'topic_num'] = X['topic_num'].astype('category')
  X['topic_num'] = pd.Categorical(X.topic_num)
  print('Types: ')
  print(X.dtypes)
  print(X['topic_num'])
    
  bins = [[0,5],[5,10],[10,20],[20,50],[50,100],[100,200],[200,500],[500,1000],[1000,10000],[10000,1000000]]
  model_lgbm = lgbm.Booster(model_file='./insight_flask_app/lgbm_model') 
  
  
  predicted_claps_class = int(np.argmax(model_lgbm.predict(X),axis=1))
  print('Predicted class: ',predicted_claps_class)
  min_claps = bins[predicted_claps_class][0]
  max_claps = bins[predicted_claps_class][1]

  perturb_df = pd.DataFrame(columns = ['feature','minperturb','dir','max_claps','perturb'])
  perturb_df['feature'] = ['compound','compound',
                     'word_count','word_count',
                     'sentence_count','sentence_count',
                     'average_sentence_length','average_sentence_length',
                     'num_images','num_images',
                    ]

  perturb_df['minperturb'] = [0.01,-0.01,
                     50,-50,
                     1,-1,
                     1,-1,
                     1,-1,
                    ]

  perturb_df['dir'] = ['make this article more positive','make this article less positive',
                     'increase the total word count','decrease the total word count',
                     'increase the total number of sentences','increase the total number of sentences',
                     'increase the average sentence length','decrease the average sentence length',
                     'increase the number of images','decrease the number of images',
                    ]
                      
  for ind,feature in enumerate(perturb_df['feature']):
    perturb_df.loc[ind,'perturb'], perturb_df.loc[ind,'max_claps'] = find_min_perturb(
                      model_lgbm, X, bins, feature, perturb_df['minperturb'][ind], max_claps)
    print('found:')
    print(ind,feature,perturb_df.loc[ind,'perturb'],perturb_df.loc[ind,'max_claps'])
#   
  indmax = np.argmax(perturb_df['max_claps'])
  max_claps_inst = perturb_df['max_claps'][indmax]
  if max_claps_inst > 0:
    instruction = perturb_df['dir'][indmax]
  else:
    instruction = ' '
    
  if 'word' in instruction:
    instruction = instruction + ' by ' + str(int(perturb_df['perturb'][indmax])) + ' words'
  elif 'sentences' in instruction:
    instruction = instruction + ' by ' + str(int(perturb_df['perturb'][indmax]))
  elif 'average' in instruction:
    instruction = instruction + ' by ' + str(int(perturb_df['perturb'][indmax])) + ' words'
  elif 'images' in instruction:
    instruction = instruction + ' by ' + str(int(perturb_df['perturb'][indmax]))

  perturb_fb,max_claps_fb = find_min_perturb(model_lgbm, X, bins, 'facebook_shares', 1, max_claps)

  return min_claps, max_claps, instruction, max_claps_inst, max_claps_fb, perturb_fb
 