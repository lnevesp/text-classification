#import required packages
#basics
import pandas as pd
import numpy as np

#misc
import gc
import time
import warnings
#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pyLDAvis.gensim
#nlp
import string
import re     #for regex
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary


#Modeling
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy import sparse


#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")

#constants
eng_stopwords = set(stopwords.words("english"))
#settings
warnings.filterwarnings("ignore")
lem = WordNetLemmatizer()
tokenizer=ToktokTokenizer()

start_time=time.time()
#importing the dataset
train=pd.read_csv("./input/train.csv")
test=pd.read_csv("./input/test.csv")
end_import=time.time()
print("Time till import:",end_import-start_time,"s")

#to seperate sentenses into words
def preprocess(comment):
    """
    Function to build tokenized texts from input comment
    """
    return gensim.utils.simple_preprocess(comment, deacc=True, min_len=3)

#tokenize the comments
train_text=train.comment_text.apply(lambda x: preprocess(x))
test_text=test.comment_text.apply(lambda x: preprocess(x))
all_text=train_text.append(test_text)
end_preprocess=time.time()
print("Time till pre-process:",end_preprocess-start_time,"s")

#checks
print("Total number of comments:",len(all_text))
print("Before preprocessing:",train.comment_text.iloc[30])
print("After preprocessing:",all_text.iloc[30])

#Phrases help us group together bigrams :  new + york --> new_york
bigram = gensim.models.Phrases(all_text)

#check bigram collation functionality
bigram[all_text.iloc[30]]


def clean(word_list):
    """
    Function to clean the pre-processed word lists

    Following transformations will be done
    1) Stop words removal from the nltk stopword list
    2) Bigram collation (Finding common bigrams and grouping them together using gensim.models.phrases)
    3) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
    """
    # remove stop words
    clean_words = [w for w in word_list if not w in eng_stopwords]
    # collect bigrams
    clean_words = bigram[clean_words]
    # Lemmatize
    clean_words = [lem.lemmatize(word, "v") for word in clean_words]
    return (clean_words)

#check clean function
print("Before clean:",all_text.iloc[1])
print("After clean:",clean(all_text.iloc[1]))

#scale it to all text
all_text=all_text.apply(lambda x:clean(x))
end_clean=time.time()
print("Time till cleaning corpus:",end_clean-start_time,"s")

#create the dictionary
dictionary = Dictionary(all_text)
print("There are",len(dictionary),"number of words in the final dictionary")

#convert into lookup tuples within the dictionary using doc2bow
print(dictionary.doc2bow(all_text.iloc[1]))
print("Wordlist from the sentence:",all_text.iloc[1])
#to check
print("Wordlist from the dictionary lookup:",
      dictionary[21],dictionary[22],dictionary[23],dictionary[24],dictionary[25],dictionary[26],
      dictionary[27])

#scale it to all text
corpus = [dictionary.doc2bow(text) for text in all_text]
end_corpus=time.time()
print("Time till corpus creation:",end_clean-start_time,"s")

#create the LDA model
ldamodel = LdaModel(corpus=corpus, num_topics=15, id2word=dictionary)
end_lda=time.time()
print("Time till LDA model creation:",end_lda-start_time,"s")

#creating the topic probability matrix
topic_probability_mat = ldamodel[corpus]

#split it to test and train
train_matrix=topic_probability_mat[:train.shape[0]]
test_matrix=topic_probability_mat[train.shape[0]:]

del(topic_probability_mat)
del(corpus)
del(all_text)
gc.collect()

#convert to sparse format (Csr matrix)
train_sparse=gensim.matutils.corpus2csc(train_matrix)
test_sparse=gensim.matutils.corpus2csc(test_matrix)
end_time=time.time()
print("total time till Sparse mat creation",end_time-start_time,"s")

# %matplotlib inline
#
# import pandas as pd, numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#
# #
# # class PreProcessing:
# #
# #     def __init__(self, input='./input/', output='./output/', train='train.csv', test='test.csv'):
# #         train = pd.read_csv(input + train)
# #         test = pd.read_csv(input + test)
#
#
#
#
#
#
# # Load Data
# train = pd.read_csv('./input/train.csv')
# test = pd.read_csv('./input/test.csv')
# subm = pd.read_csv('./input/sample_submission.csv')
#
# print(train.head())
#
# lens = train.comment_text.str.len()
# lens.mean(), lens.std(), lens.max()
#
# lens.hist();
#
# label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# train['none'] = 1-train[label_cols].max(axis=1)
# train.describe()
#
# len(train),len(test)
#
#
# COMMENT = 'comment_text'
# train[COMMENT].fillna("unknown", inplace=True)
# test[COMMENT].fillna("unknown", inplace=True)
#
# import re, string
# re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
# def tokenize(s):
#     return re_tok.sub(r' \1 ', s).split()
#
#
# n = train.shape[0]
# vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                smooth_idf=1, sublinear_tf=1 )
# trn_term_doc = vec.fit_transform(train[COMMENT])
# test_term_doc = vec.transform(test[COMMENT])
#
# trn_term_doc, test_term_doc
#
# def pr(y_i, y):
#     p = x[y==y_i].sum(0)
#     return (p+1) / ((y==y_i).sum()+1)
#
# x = trn_term_doc
# test_x = test_term_doc
#
# def get_mdl(y):
#     y = y.values
#     r = np.log(pr(1,y) / pr(0,y))
#     m = LogisticRegression(C=4, dual=True)
#     x_nb = x.multiply(r)
#     return m.fit(x_nb, y), r
#
# preds = np.zeros((len(test), len(label_cols)))
#
# for i, j in enumerate(label_cols):
#     print('fit', j)
#     m,r = get_mdl(train[j])
#     preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
#
# submid = pd.DataFrame({'id': test["id"]})
# submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
# submission.to_csv('./output/submission.csv', index=False)