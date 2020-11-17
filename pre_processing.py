#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:18:01 2020

@author: a.gogohia
"""

import json
import pandas as pd
import nltk
import de_core_news_md # spacy german language model
import re
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


##### REMOVE NOISE #####

def remove_emails(sentences):
    """Remove all emails from list/series of sentences"""
    new_sentences = [re.sub('\S*@\S*\s?', '', sentence) for sentence in sentences]
    return new_sentences

def remove_new_line_chars(sentences):
    """Remove all new line characters from list/series of sentences"""
    new_sentences = [re.sub('\s+', '', sentence) for sentence in sentences]
    return new_sentences

def remove_single_quote_chars(sentences):
    """Remove all single quote characters from list/series of sentences"""
    new_sentences = [re.sub('\'', '', sentence) for sentence in sentences]
    return new_sentences

def remove_numbers(sentences):
    """Remove all numbers from text"""
    new_sentences = [re.sub(r'[0-9]+', '', sentence) for sentence in sentences]
    return new_sentences

def remove_punctuation(sentences):
    """Remove punctuation from text"""
    new_sentences = [re.sub(r'[^\w\s]+', '', sentence) for sentence in sentences]
    return new_sentences

def remove_noise(data):
    data = remove_emails(data)
    data = remove_single_quote_chars(data)
    data = remove_numbers(data)
    data = remove_punctuation(data)
    return data

##### NORMALIZE #####

def to_lowercase(sentences):
    """Convert all characters to lowercase from list/series of sentences"""
    new_sentences = [sentence.lower() for sentence in sentences]
    return new_sentences

def remove_stopwords(sentences):
    """Remove stopwords from list/series of sentences"""
    new_sentences = [[word for word in doc.split() if word not in stopwords] for doc in sentences]
    return new_sentences

def lemmatize(sentences, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Lemmatize all words i.e. gefunden -> finden"""
    texts_out = []
    nlp = de_core_news_md.load()
    for sent in sentences:
        doc = nlp(sent) 
        texts_out.append(' '.join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def normalize(data):
    data = lemmatize(data)
    data = to_lowercase(data)
    data = remove_stopwords(data)
    return data

##### TOKENIZE #####

def tokenize(sentences):
    new_sentences = [sentence.split() for sentence in sentences]
    return new_sentences

##### CREATE DICTIONARY AND CORPUS #####

def create_bow(texts, verbose=True):
    id2word = gensim.corpora.Dictionary(texts)
    freq_before = len(id2word)
    id2word.filter_extremes(no_below=1, no_above=0.9, keep_n=100000)
    corpus = [id2word.doc2bow(text) for text in texts]
    if verbose == True:
        print('### Filtering least and most occuring tokens ###')
        print(f'{freq_before} tokens before filtering -> {len(id2word)} after filtering')
    return id2word, corpus

def create_tfidf(texts):
    id2word, corpus = create_bow(texts)
    tfidf = gensim.models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
    return id2word, tfidf_corpus


if __name__ == '__main__':
    
    # READ DATA
    data = pd.read_csv('data/articles.csv', sep='\n', header=None)
    data = data[0]
    
    stopwords = nltk.corpus.stopwords.words('german')
    
    # PRE-PROCESS DOCUMENTS
    processed_data = remove_noise(data)
    processed_data = normalize(processed_data)
    # processed_data = tokenize(processed_data) # not needed since normalize() function returns tokens

    # SAVE FILE
    with open('processed_data.txt', 'w') as f:
        json.dump(processed_data, f)
