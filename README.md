# doc_clustering
Clustering using tfidf and bert and Opitics on documents (finding similar clauses)

Idea here is to use clustering to put togather similar sentences, so we will have clusters with the clauses(where all the similar will be together). We can manully filter-out these out. This solution can furthher refined by better text cleaning and preprocessing.
I uesd OPTICS(with min elements = 3) instead of the k-means so that we do not have to choose the number of clusters


**TO USE THE STANFORD LIB**
1. download the stanfordCoreNLP from stanford site or do following:(link: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK)
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip
cd stanford-corenlp-full-2018-02-27
2. use following command to run the stanford server:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 & 
3. then run the solution file to run the code

Solution:
1. Breaked the doc into sentences on the basis of the splitsentence( ).
2. Cleaned the doc a bit using standerd libs
3. User stanford tree parser to get tree structure of the sentences
4. Extracted sentences which had ADJP (Adjective phrases in them)
5. Used these extrated clauses for clustering
6. There were 2 representations used a) TF-IDF and b) BERT
7. OPTICS clustering(modified version of the DBSCAN) for clustering to make the output cluster numbers dynamic
8. Results were saved in **result.json**

Result comparison:
There are 2 files result_tfidf.json and result_bert.json which contains the results of both approach, an by manually looking at the results it can directly be seen that the results of the BERT representation is better.
**IMPORTANT: The results can further be refined using breaking big paragraphs in smaller sentences.**

How to use BERT embeddings:
I used following opensource github project to set up bert embeddings for the solution:
1. Clone or download the git repo: https://github.com/UKPLab/sentence-transformers#Training
2. Use the instractions to install the transformers by pip install -U sentence-transformers or pip install -e . (using source)
3. Copy the content of this directory in the home folder of the above repo and execute the code.
4. You can use https://jsoneditoronline.org/ to visulize the results form result.json

install the spacy en packedge: python -m spacy download en

**Running the code**:
use the file **legal_clustering.py**

**using tff-idf**: uncomment: clusters = getClusters(line_list) (and comment: clusters = getBertClusters(line_list)) (line:195)
**using bert**: uncomment: clusters = getBertClusters(line_list) (and comment: clusters = getClusters(line_list)) (line: 194)

Run the code by using: *python3 legal_clustering.py*

use the **python3** to run the code and download the depedencies

**If you have trouble using bert embedings(it will download the firstime), follow these instructions from the mentioned git-repo:**

**Clustering**:
**examples/application_clustering.py** depicts an example to cluster similar sentences based on their sentence embedding similarity.

As before, we first compute an embedding for each sentence:

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences

corpus = ['A man is eating a food.',
          'A man is eating a piece of bread.',
          'A man is eating pasta.',
          'The girl is carrying a baby.',
          'The baby is carried by the woman',
          'A man is riding a horse.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'Someone in a gorilla costume is playing a set of drums.',
          'A cheetah is running behind its prey.',
          'A cheetah chases prey on across a field.']

corpus_embeddings = embedder.encode(corpus)
Then, we perform k-means clustering using scipy:

# Perform kmean clustering
num_clusters = 5
whitened_corpus = scipy.cluster.vq.whiten(corpus_embeddings)
code_book, _ = scipy.cluster.vq.kmeans(whitened_corpus, num_clusters)
cluster_assignment, _ = scipy.cluster.vq.vq(whitened_corpus, code_book)
The output looks like this:

Cluster  1
['A man is riding a horse.', 'A man is riding a white horse on an enclosed ground.']

Cluster  2
['A man is eating a food.', 'A man is eating a piece of bread.', 'A man is eating pasta.']

Cluster  3
['A monkey is playing drums.', 'Someone in a gorilla costume is playing a set of drums.']

Cluster  4
['The girl is carrying a baby.', 'The baby is carried by the woman']

Cluster  5
['A cheetah is running behind its prey.', 'A cheetah chases prey on across a field.']
