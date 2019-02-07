import numpy as np
import pandas as pd
import random

from spacy.lang.en import English
parser = English()

import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
import nltk.data
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
import textstat

from langdetect import detect
from langdetect import DetectorFactory 
DetectorFactory.seed = 0 

import pickle

import seaborn as sns
import matplotlib.pyplot as plt


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def prepare_text_for_lda(text):
    re_tokenizer = RegexpTokenizer(r'\w+')
    tokens = re_tokenizer.tokenize(text.lower())
    newStopWords = ['company','business','startup','http','com','www','https','will']
    en_stop = nltk.corpus.stopwords.words('english')
    en_stop.extend(newStopWords)

    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [get_lemma(token) for token in tokens]
    return tokens
    

print('Reading in dataframes...')
columns = ['url','title','author','text','claps','reading_time','num_images']
df_startuptag = pd.DataFrame(columns = columns)

columns_fb = ['url','facebook_shares']
df_fb = pd.DataFrame(columns = columns_fb)

years = ['2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
days = ['01','02','03','04','05','06','07','08','09','10',
  '11','12','13','14','15','16','17','18','19','20',
  '21','22','23','24','25','26','27','28','29','30','31',]
  
for year in years:
    for month in months:
        for day in days:
            file = '../Data/medium_startuptag_stories_'+str(month)+str(day)+str(year)+'_v2.pkl'
            df_test = pd.read_pickle(file)
            df_startuptag = df_startuptag.append(df_test, sort=False)
            
            file = '../Data/medium_startuptag_fb_shares_'+str(month)+str(day)+str(year)+'_v1.pkl'
            df_test = pd.read_pickle(file)
            df_fb = df_fb.append(df_test, sort=False)
            
df_startuptag.set_axis(['urls','title','author','text','claps','reading_time','num_images'],axis=1,inplace=True)

df_fb.set_axis(['urls','facebook_shares'],axis=1,inplace=True)
df_fb['facebook_shares'] = pd.to_numeric(df_fb['facebook_shares'])

df = df_startuptag.merge(df_fb, on='urls')
df = df.dropna(axis=0)
df.reset_index(inplace=True)
df = df.drop(['index'],axis=1)
print('Number of articles with full data: ',len(df))

print('Reformatting data types...')
df['claps_num'] = np.nan
for ind,claps in enumerate(df.claps):
    if 'K' in claps:
        claps_num = 1000*pd.to_numeric(claps.replace('K',''))
    else:
        claps_num = pd.to_numeric(claps)
    df.loc[ind,'claps_num'] = claps_num

df['reading_time_num'] = np.nan
for ind,reading_time in enumerate(df.reading_time):
    reading_time_num = pd.to_numeric(reading_time.replace(' min read',''))
    df.loc[ind,'reading_time_num'] = reading_time_num

df['num_images'] = pd.to_numeric(df['num_images'])

print('Reformatting titles...')
df['title'] = df.title.str.replace('– The Startup – Medium','')
df['title'] = df.title.str.replace('– Medium','')
df['title'] = df.title.str.replace('– The Startup','')
df['text'] = df.text.str.replace('\n',' ')

print('Tokenizing titles and text...')
df['title_tokens'] = df.apply(lambda x: prepare_text_for_lda(x['title']), axis=1)
df['text_tokens'] = df.apply(lambda x: prepare_text_for_lda(x['text']), axis=1)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
df['text_sentences'] = df.apply(lambda x: sent_detector.tokenize(x['text']), axis=1)

for ind,text in enumerate(df.text):
    if not text:
        df.loc[ind,'text_tokens'] = np.nan
    else:
        if detect(text)!='en':
            df.loc[ind,'text_tokens'] = np.nan
        
df = df.dropna(axis=0)
df.reset_index(inplace=True)
df = df.drop(['index'],axis=1)

print('Calculating writing-style features...')
df['readability_index'] = df.apply(lambda x: textstat.text_standard(x['text'], float_output=True), axis=1)
re_tokenizer = RegexpTokenizer(r'\w+')

df['word_count'] = np.nan
df['unique_word_count'] = np.nan
df['average_word_length'] = np.nan
df['sentence_count'] = np.nan
df['average_sentence_length'] = np.nan

for ind,sents in enumerate(df.text_sentences):
    num = len(sents)
    num_tokens = 0
    num_characters = 0
    unique_tokens = []
    for sent in sents:
        num_tokens += len(re_tokenizer.tokenize(sent.lower()))
            
        for word in re_tokenizer.tokenize(sent.lower()):
            num_characters += len(word)
            if word not in unique_tokens:
                unique_tokens.append(word)
                
    df.loc[ind,'sentence_count'] = num
    df.loc[ind,'word_count'] = num_tokens
    df.loc[ind,'unique_word_count'] = len(unique_tokens)
    df.loc[ind,'average_sentence_length'] = num_tokens/num
    df.loc[ind,'average_word_length'] = num_characters/num_tokens

print('Performing topic modeling...')
text_data = df.text_tokens
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
dictionary.save('../Model/dictionary.gensim')

num_topics = 20
lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary, 
                                      alpha='auto', eta='auto', passes=50)
lda.save('../Model/lda.gensim')

df['topic_num'] = np.nan
for ind,tokens in enumerate(df['text_tokens']):
    test = lda.get_document_topics(dictionary.doc2bow(tokens))
    topics = sorted(test,key=lambda x:x[1],reverse=True)
    df.loc[ind,'topic_num'] = topics[0][0]

print('Performing sentiment analysis...')
analyser = SentimentIntensityAnalyzer()

df['compound'] = np.nan
df['pos'] = np.nan
df['neg'] = np.nan
df['neu'] = np.nan

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

for ind,sents in enumerate(df.text_sentences):
    compound = []
    pos = []
    neg = []
    neu = []
    for sent in sents:
        score = analyser.polarity_scores(sent)
        compound.append(score['compound'])
        pos.append(score['pos'])
        neg.append(score['neg'])
        neu.append(score['neu'])
    df.loc[ind,'compound'] = np.mean(compound)
    df.loc[ind,'pos'] = np.mean(pos)
    df.loc[ind,'neg'] = np.mean(neg)
    df.loc[ind,'neu'] = np.mean(neu)

df = df.dropna(axis=0)
df.reset_index(inplace=True)
df = df.drop(['index'],axis=1)

file = '../Model/df_medium_preprocessed.pkl'
df.to_pickle(file)
print('Done!')
