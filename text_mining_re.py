# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:57:05 2023

@author: SHRI
"""

import re
sentence5="sharat twitted , Wittnessing 70th republic day India  from Rajpath, \new-Delhi , memorizing performance by Indian Army!"
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()
# extracting n-grams
#n-gram can be extracted using three technique
#1 cluster defined function
#2 NLTK
#3 TextBlob

###################################
#extracting n-grams using custom defined function

import re
def n_gram_extractor(input_str,n):
    tokens =re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
        

n_gram_extractor("The cute little boy is playing with kitten",2)
n_gram_extractor("The cute little boy is playing with kitten",3)

#----------------------------------------------------------------------------------------------------
import nltk
from nltk import ngrams
#extracting n-grams with nltk
list(ngrams("The cute little boy is playing with kitten",2))
list(ngrams("The cute little boy is playing with kitten",3))

#----------------------------------------------------------------------------------------------------

from textblob import TextBlob
blob=TextBlob("The cute little boy is playing with kitchen.")
blob.ngrams(n=2)
blob.ngrams(n=3)

#----------------------------------------------------------------------------------------------------
##Tokenization using keras
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)
blob.words

#----------------------------------------------------------------------------------------------------
#Tokenizer using TextBlob
from textblob import TextBlob
blob =TextBlob(sentence5)
blob.words

#----------------------------------------------------------------------------------------------------
## Tweet Tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)

#----------------------------------------------------------------------------------------------------

from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace('!', ' ').split())

# multi word expression

from nltk.tokenize import MWETokenizer
sentence5
mwe_tokenizer=MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace(',', ' ').split())

#----------------------------------------------------------------------------------------------------
#Regular Expression TOkennizer
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\S+')
reg_tokenizer.tokenize(sentence5)

#----------------------------------------------------------------------------------------------------
##White Space tokenizer
from nltk.tokenize import WhitespaceTokenizer
wh_tokenize=WhitespaceTokenizer()
wh_tokenize.tokenize(sentence5)

#----------------------------------------------------------------------------------------------------
from nltk.tokenize import WordPunctTokenizer
wp_tokenize=WordPunctTokenizer()
wp_tokenize.tokenize(sentence5)

#----------------------------------------------------------------------------------------------------
sentence6="I Love playing cricket. Cricket players practices hard in their inning"
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())

#----------------------------------------------------------------------------------------------------
sentence7="Before eating ,it would be nice to sanitize your hands with a sanitizer"
from nltk.stem.porter import PorterStemmer
ps_stemmer =PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])

#----------------------------------------------------------------------------------------------------
#Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
Lemmatizer=WordNetLemmatizer()
sentence8="The codes executed todays are far better than what  executed "
words=word_tokenize(sentence8)
" ".join([Lemmatizer.lemmatize(word) for word in words])

#----------------------------------------------------------------------------------------------------
#singularize and pluaralization 
from textblob import TextBlob
sentence9=TextBlob("She sells seashells on the seashare")
words=sentence9.words
##we want to make word[2] .i.e seashells in singular form
sentence9.words[2].singularize()
## we want word 5 i.e seashare in plural form
sentence9.words[5].pluralize()

#----------------------------------------------------------------------------------------------------

#language translation from spanish to English
from textblob import  TextBlob
en_blob =TextBlob(u'muy bien')
en_blob.translate(from_lang='es', to='en')
#es:spanish en:English


#----------------------------------------------------------------------------------------------------
##custom stopwords removal
from nltk import word_tokenize
sentence9="She sells seashells on the seashare"
custom_stop_word_list=['she','on','the','am','is']
words=word_tokenize(sentence9)
" ".join([word for word in words if word.lower() not in custom_stop_word_list])
#select the words which are not in defined list.

#----------------------------------------------------------------------------------------------------

#extracting the general feature from the raw text
#number of words
#detect the presence of wh words
#polarity
#subjectivity
#language identification
#-----------------------------------------------------------------------------------------------------
#To identify the number of words
import pandas as pd
df=pd.DataFrame([['The vaccine for covid-19 will be announced on 1st August'],
                 ['Do you know how much expectation the world population is having from this research?'],
                 ['The risk of virus will come to an end on 31st July']])
df.columns=['Text']
df

#now let us measure the number of words
from textblob import TextBlob
df['number_of_words']=df['Text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words']

#Detect the presence of words wh
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['Text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

#Polarity of the sentence(rating in amazon or flipkart product)
df['polarity']=df['Text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
sentence10="I like fantastic example and I lite it very much"
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This is fantastic example and I like it very much"
pol=TextBlob(sentence10).sentiment.polarity
pol


sentence10="This was helpful example but I would have prefer another "
pol=TextBlob(sentence10).sentiment.polarity
pol

sentence10="This is my personal opinion that it was helpful example but I would prefer another one"
pol=TextBlob(sentence10).sentiment.polarity
pol

#-----------------------------------------------------------------------------------------------------
#Subjectivity of the dataframe df and check wheather there is personal opinion 
df['subjectivity']=df['Text'].apply(lambda  x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

#-----------------------------------------------------------------------------------------------------
## TO find the language of the sentence , this part of code will get http error
df['Language']=df['Text'].apply(lambda x:TextBlob(str(x)).detect_language())
df['Language']=df['Text'].apply(lambda x:TextBlob(str(x)).detect_language())













