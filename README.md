# topic-modelling-lda
A setup for topic modelling with LDA on a German dataset.

## Dataset
The data that was used for the topic modelling task:
https://github.com/tblock/10kGNAD

The dataset contains 10k news articles from an Austrian online news platform. Each article is assigned to one specific category (out of 9).
Since I am using an unsupervised topic moelling method here, I will only be using the article texts.

Before pre-processing the data I removed the classification tags and proceeded with the texts only.

## Pre-processing
In the pre-processing step I am removing noise first, before I normalize the text by applying lemmatization, transforming to lower cases and removing stopwords. 
After tokenizing the text, one can choose to create either a bag of words or a tf-idf-model for further processing.
In my experiments I consitently ended up with higher coherence values using the bag of words method.

## Model
In this setup I am using the LDA model only. 
To find the optimal model, I loop over a range of possible topic numbers and choose the one with the highest coherence score.
