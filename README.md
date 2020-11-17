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

## Results for the best setup

'''shell
2020-11-17 14:41:49,211 : INFO : topic #0 (2.939): 0.013*"gut" + 0.011*"jahr" + 0.011*"groß" + 0.010*"mensch" + 0.009*"immer" + 0.008*"geben" + 0.008*"schon" + 0.008*"finden" + 0.007*"gehen" + 0.007*"zeigen"
2020-11-17 14:41:49,212 : INFO : topic #1 (1.785): 0.019*"erst" + 0.011*"tier" + 0.010*"weit" + 0.008*"erste" + 0.007*"kommen" + 0.007*"rund" + 0.007*"bereits" + 0.006*"ende" + 0.006*"tag" + 0.006*"bank"
2020-11-17 14:41:49,214 : INFO : topic #2 (0.050): 0.036*"handel" + 0.024*"hund" + 0.017*"satz" + 0.011*"sammlung" + 0.010*"mitgliedsstaaten" + 0.008*"schlange" + 0.008*"fix" + 0.007*"korruption" + 0.007*"spielraum" + 0.006*"studentin"
2020-11-17 14:41:49,215 : INFO : topic #3 (0.026): 0.018*"sprecherin" + 0.016*"zurückkehren" + 0.013*"franke" + 0.006*"presseagentur" + 0.003*"bündeln" + 0.003*"limit" + 0.001*"inoffiziell" + 0.000*"vizebürgermeisterin" + 0.000*"negativzinsen" + 0.000*"fremdwährungskredite"
2020-11-17 14:41:49,216 : INFO : topic #4 (0.953): 0.009*"regierung" + 0.008*"präsident" + 0.007*"griechische" + 0.007*"angabe" + 0.007*"berichten" + 0.006*"fund" + 0.006*"mensch" + 0.006*"menschlich" + 0.006*"staat" + 0.006*"artikel"
2020-11-17 14:41:49,217 : INFO : topic #5 (0.160): 0.036*"erkenntnis" + 0.022*"satellit" + 0.018*"version" + 0.012*"decken" + 0.012*"folgend" + 0.012*"regen" + 0.010*"typisch" + 0.010*"aufheben" + 0.009*"anlegen" + 0.008*"star"
2020-11-17 14:41:49,218 : INFO : topic #6 (2.080): 0.044*"prozent" + 0.033*"euro" + 0.029*"jahr" + 0.020*"forscher" + 0.019*"million" + 0.014*"hoch" + 0.012*"rund" + 0.012*"milliarde" + 0.012*"studie" + 0.010*"unternehmen"
2020-11-17 14:41:49,219 : INFO : topic #7 (0.080): 0.043*"stern" + 0.035*"krankheit" + 0.028*"reduzieren" + 0.014*"auszeichnung" + 0.013*"italienisch" + 0.012*"farbe" + 0.008*"künstliche" + 0.007*"mehrheitlich" + 0.007*"tragisch" + 0.007*"krebs"
2020-11-17 14:41:49,221 : INFO : topic #8 (0.008): 0.022*"welle" + 0.004*"reformieren" + 0.000*"quantenmechanik" + 0.000*"quantenphysik" + 0.000*"photon" + 0.000*"gravitationswellen" + 0.000*"elektron" + 0.000*"quantenoptik" + 0.000*"gedankenexperiment" + 0.000*"spalte"
2020-11-17 14:41:49,222 : INFO : topic #9 (4.994): 0.013*"sagen" + 0.011*"sollen" + 0.011*"geben" + 0.008*"weit" + 0.007*"land" + 0.007*"neu" + 0.006*"etwa" + 0.006*"neue" + 0.006*"bereits" + 0.006*"müssen"
'''
'''shell
The coherence score for the single model is 0.4015469738237324.
'''
