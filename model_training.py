#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:11:24 2020

@author: a.gogohia
"""


import json
import gensim
from pre_processing import (create_bow,
                            create_tfidf)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

##### CREATE TOPIC MODELS #####
    
def create_lda_model(corpus, id2word, num_topics=10, passes=10):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           update_every=1,
                                           chunksize=100,
                                           passes=passes,
                                           alpha='auto',
                                           per_word_topics=True)
    return lda_model
    
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
                                       id2word=dictionary, passes=10)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts,
                                                      dictionary=dictionary,
                                                      coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

if __name__ == '__main__':
    
    # READ PRE-PROCESSED DATA
    with open('processed_data.txt', 'r') as f:
        processed_data = json.load(f)
        
    # CREATE DICTIONARY AND CORPUS
    id2word, corpus = create_bow(processed_data) # bag of words
    # id2word, tfidf_corpus = create_tfidf(processed_data) # tf-idf
    
    # CHECK FREQUENT TOKENS
    dfs_desc = sorted(id2word.dfs.items(), key=lambda t: t[1], reverse=True)
    
    print('### Most Frequent Tokens in', id2word.num_docs, 'Documents ###')
    for (k,v) in dfs_desc[0:10]: print('{freq}: {token}'.format(token=id2word[k], freq=v))
    
    print('### Least Frequent Tokens in', id2word.num_docs, 'Documents ###')
    for (k,v) in dfs_desc[-10:]: print('{freq}: {token}'.format(token=id2word[k], freq=v))

    # TRAIN SINGLE MODEL
    lda_model = create_lda_model(corpus=corpus, id2word=id2word, num_topics=10)
    # OR 
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                        id2word=id2word,
    #                                        num_topics=10, 
    #                                        random_state=100,
    #                                        update_every=1,
    #                                        chunksize=100,
    #                                        passes=10,
    #                                        alpha='auto',
    #                                        per_word_topics=True)

    # PRINT TOPICS
    lda_model.print_topics()

    # COMPUTE COHERENCE SCORE
    coherence_score = gensim.models.CoherenceModel(model=lda_model, texts=processed_data, dictionary=id2word, coherence='c_v').get_coherence()
    print(f'### The coherence score for the single model is {coherence_score}. ###')

    # ITERATE OVER POSSIBLE NUMBER OF TOPICS
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=processed_data, start=5, limit=50, step=5)

    
    
    
    
    