import time
import docx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

import nltk
nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
import json


stop_words = stopwords.words('english')


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def preProcess(data):
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    #Remove tab spaces
    # data = [re.sub(r'\s', '', str(sent)) for sent in data]

    #Remove queries with 3 or less words
    data = [sent for sent in data if len(sent.split(' ')) > 3]

    #Remove mails
    data = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data]

    #Remove special characters and digits

    data = [re.sub("(\\d|\\W)+"," ",str(sent)) for sent in data]

    #Remove new line characters
    data = [re.sub('\s+', ' ', str(sent)) for sent in data]

    #Remove urls
    data = [re.sub(r'https?://\S+|www\.\S+','',str(sent)) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    data_words = list(sent_to_words(data))  
    # Build the bigram and trigram models
    # bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    # data_words_bigrams = make_bigrams(data_words_nostops)

    

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized

def cluster_dictionary(doc,cluster_labels):
	labels = np.unique(cluster_labels)
	clustering_dict = dict()
	for label in labels:
		result = np.where(cluster_labels == label)
		indices = result[0]
		clustering_dict[label] = []
		for index in indices:
			clustering_dict[label].append(doc[index])
	del clustering_dict[-1]
	return clustering_dict


def getClusters(doc):
	processed_doc = [' '.join(text) for text in preProcess(doc)]
	tfidf = TfidfVectorizer()
	vectorizer = tfidf.fit_transform(processed_doc)
	# clusters = dbscan_clustering(doc, vectorizer, 0.75,3)
	clusters = optics_clustering(doc,vectorizer)
	final_clusters = {}

	# Removing duplicacy in clusters
	for key in clusters:
		final_clusters[str(key)] = list(set(clusters[key]))

	return final_clusters


def optics_clustering(doc,vectorizer):
	cluster = OPTICS(min_samples=3).fit(vectorizer.toarray())
	return cluster_dictionary(doc,cluster.labels_)


def dbscan_clustering(doc,vectorizer,epsilon,min_samples):
	cluster = DBSCAN(eps=epsilon, min_samples=min_samples).fit(vectorizer)
	return cluster_dictionary(doc,cluster.labels_)


def ch_format(dict_):
	intent_data = []
	i=0
	for topic,intent in dict_.items():
		temp = {}
		temp['intent_name'] = topic
		temp['intent_number'] = i
		temp['text'] = intent
		temp['response'] = ""
		i=i+1
		intent_data.append(temp)
	return intent_data


def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


if __name__ == "__main__":
	start_time = time.time()

	print("-- starting the processing --")

	filename = "legal_file.docx"

	data = getText(filename)

	## splitting in lines
	line_list = data.splitlines( )

	clusters = getClusters(line_list)
	r = json.dumps(clusters)
	clusters_json = json.loads(r)
	
	with open('result.json', 'w') as fp:
		json.dump(clusters, fp)

	#print(clusters_json)

	print("--- %s seconds ---" % (time.time() - start_time))


