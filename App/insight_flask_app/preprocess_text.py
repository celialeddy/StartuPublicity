import numpy as np
import pandas as pd
from spacy.lang.en import English
parser = English()

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
import textstat

def get_lemma(word):
  lemma = wn.morphy(word)
  if lemma is None:
    return word
  else:
    return lemma

def prepare_text_for_lda(text):
  re_tokenizer = RegexpTokenizer(r'\w+')
  tokens = re_tokenizer.tokenize(text.lower())
  newStopWords = ['company','business','startup','http','com','www','https','will',
      'follow','following','blocked','unblock']  
  en_stop = nltk.corpus.stopwords.words('english')
  en_stop.extend(newStopWords)

  tokens = [token for token in tokens if token not in en_stop]
  tokens = [token for token in tokens if not token.isdigit()]
  tokens = [get_lemma(token) for token in tokens]
  return tokens
        
def preprocess_text(title, text, num_images):
  df = pd.DataFrame(columns=['title','text','num_images'])
  df.loc[0,'title'] = title.replace('– The Startup – Medium','')
  df['title'] = df.title.str.replace('– Medium','')
  df['title'] = df.title.str.replace('– The Startup','')
  df.loc[0,'text'] = text.replace('\n',' ')
  df.loc[0,'num_images'] = num_images
  df['num_images'] = pd.to_numeric(df['num_images'])
  df['facebook_shares'] = 0
  df['title_tokens'] = df.apply(lambda x: prepare_text_for_lda(x['title']), axis=1)
  df['text_tokens'] = df.apply(lambda x: prepare_text_for_lda(x['text']), axis=1)
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  df['text_sentences'] = df.apply(lambda x: sent_detector.tokenize(x['text']), axis=1)
      
  df['readability_index'] = df.apply(lambda x: textstat.text_standard(x['text'], float_output=True), axis=1)
  re_tokenizer = RegexpTokenizer(r'\w+')
  
  num_tokens = 0
  num_characters = 0
  unique_tokens = []
  for sent in df.text_sentences[0]:
    num_tokens += len(re_tokenizer.tokenize(sent.lower()))
    for word in re_tokenizer.tokenize(sent.lower()):
      num_characters +- len(word)
      if word not in unique_tokens:
        unique_tokens.append(word)

  df['word_count'] = num_tokens
  df['unique_word_count'] = len(unique_tokens)
  df['average_word_length'] = num_characters/num_tokens
  df['sentence_count'] = len(df.text_sentences)
  df['average_sentence_length'] = num_tokens/len(df.text_sentences[0])

  lda = gensim.models.ldamodel.LdaModel.load('./insight_flask_app/lda.gensim')
  dictionary = corpora.Dictionary.load('./insight_flask_app/dictionary.gensim')
  
  tokens = df['text_tokens'][0]
  test = lda.get_document_topics(dictionary.doc2bow(tokens))
  topics = sorted(test,key=lambda x:x[1],reverse=True)
  topic_num = topics[0][0]
  df.loc[0,'topic_num'] = topics[0][0]

  analyser = SentimentIntensityAnalyzer()
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  compound = []
  pos = []
  neg = []
  neu = []
  
  for sent in df.text_sentences[0]:
    score = analyser.polarity_scores(sent)
    compound.append(score['compound'])
    pos.append(score['pos'])
    neg.append(score['neg'])
    neu.append(score['neu'])
  df['compound'] = np.mean(compound)
  df['pos'] = np.mean(pos)
  df['neg'] = np.mean(neg)
  df['neu'] = np.mean(neu)    

  min_claps_topics = [0,0,0,50,0,0,0,0,0,0,0,0,0,0,0,0,100,0,200,50]
  max_claps_topics = [5,5,5,100,5,5,5,5,5,5,5,5,5,5,5,5,200,5,500,100]
  max_max_topics = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 500, 1000, 1000,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
  min_claps_topic = min_claps_topics[int(topics[0][0])]
  max_claps_topic = max_claps_topics[int(topics[0][0])]
  max_max_topic = max_max_topics[int(topics[0][0])]
  return df, min_claps_topic, max_claps_topic, max_max_topic
  
