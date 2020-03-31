# IRDM Passage Ranking System
This is the Course Project of Information Retrieval and Data Mining. The goal is to build a passage ranking system using the skills presented in the course, which includes:
+ Text Processing & Indexing
+ Probabilistic Retrieval Models
+ Evaluation of Information Retrieval Systems
+ Page Rank & Relevance Feedback
+ Data Mining and Machine Learning Introduction
+ Neural Models for IR and DM
+ Topic Models and Vector Semantics (Embeddings)
+ Compression

The  course project contains two parts, as a beginner in the field of information retrieval, machine learning and data mining, the two parts give good learning and implementation practice experience in the above fields. 
+ The datasets used in the project are too large, which are not contained in this repository. 
## Part 1
The dataset used in this repository can be downloaded [Here](https://drive.google.com/file/d/1eKDfmDZoVuDADcR_HGMHMnjNJHDrXUs9/view)
+ Text pre-processing: word tokenisation, lowercase words, stopwords removing 
+ Inverted index:  store candidate passages in a word dictionary as entries of {word: (passage ID, word frequency in the passage)} 
+ Vector Space Model to Rank the Passages for each Query
+ BM25 Model to Rank the Passages for each Query
## Part 2
The dataset used in this repository can be downloaded [Here](https://drive.google.com/file/d/1npkPA-BdiGELHfBrUOcpqumjbQTspg9p/view)
+ Implement Metrics to Evaluate Ranking System: Average Precision and NDCG (Normalised Discounted Cumulative Gain)
+ Word Embedding use Python Spacy pre-trained model "en_core_web_md"
+ Implement Advance Models to score the relevancy of each pair of query and passage to rank the passages. These models include: - Logistic Regression; - LambdaMART, -Neural Netword Model (Tensorflow or PyTorch).
