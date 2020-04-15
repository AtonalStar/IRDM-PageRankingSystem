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
+ File Structure
``` bash
└──part1
	 ├── IRDM_2020_Assignment.pdf
	 ├── IRDM_Course_Project_Part_1_Report.pdf
	 ├── Code	  
	 │     ├── dataset (Empty)
	 │	   │	   ├── candidate_passages_top1000.tsv
	 │	   │	   ├── passage_collection.txt
	 │	   │	   └── test-queries.tsv
	 │     ├── Inverted=Index
	 │	   │	   ├── ExtractPassages. py
	 │	   │	   └── inverted_index. py
	 │     ├── invertedResult
	 │	   ├── Models
	 │	   │	   └── models. py
	 │	   ├── Text-Statistics
	 │	   │	   └── preProcess. py
	 │	   └── README.txt
	 ├── BM25.txt
	 └── VS.txt
```
## Part 2
The dataset used in this repository can be downloaded [Here](https://drive.google.com/file/d/1npkPA-BdiGELHfBrUOcpqumjbQTspg9p/view)
+ Implement Metrics to Evaluate Ranking System: Average Precision and NDCG (Normalised Discounted Cumulative Gain)
+ Word Embedding use Python Spacy pre-trained model "en_core_web_md"
+ Implement Advance Models to score the relevancy of each pair of query and passage to rank the passages. These models include: - Logistic Regression; - LambdaMART (XGBoost), -Neural Netword Model (PyTorch).
+ File Structure
``` bash
└──SubmitFolder
	  ├── IRDM Course Project Part 2 Report.pdf
	  ├── Code
	  │	    ├── part1_code_for_BM25
	  │		│		    ├── Passages (For the output of extracted passages)
	  │	    │    		├── ExtractPassages. py
	  │		│		    ├── generateBM25. py
	  │		│		    └──BM25.txt
	  │     ├── all_queries.tsv
	  │     ├── Metrics. py
	  │     ├── ModelsEvaluation. py
	  │		├── train_data_line_extract.py
	  │		├── train_data_line_1.txt
	  │		├── validList.txt (Number of valid(1.0) items per 10000 lines)
	  │		├── Embed. py
	  │		├── embed10000.tsv (Embedding of 2310000 - 2320000 train_data)[Generate from Embed. py]
	  │     ├── LR
	  │    	│	 ├── LogisticRegression. py
	  │     │    ├── generateLR. py
	  │     │    ├── LRWeight_10000.txt
	  │     │    ├── LR001.txt
	  │     │    ├── LR002.txt (LR.txt)
	  │     │    └── LR003.txt
	  │     ├── LM
	  │		│	 ├── LambdaMART. py
	  │		│	 ├── generateLM. py
	  │		│	 ├── xgb1.json
	  │		│	 ├── xgb2.json
	  │		│	 ├── xgb3json
	  │		│	 ├── xgb4json
	  │		│	 ├── LM-1.txt
	  │		│	 ├── LM-2.txt (LM.txt)
	  │		│	 ├── LM-3.txt
	  │     │	 └── LM-4.txt 
	  │     └── NN
	  │			 ├── NNmodel. py
	  │			 ├── genNN. py
	  │			 ├── nn.pkl
	  │			 └── NN.txt
	  ├── part2 (Too large to be submitted, so it's empty)
	  │     ├── train_data.tsv
	  │     └── validation_data.tsv
	  ├── Test Results				 
	  │	         ├── LR.txt
	  │	         ├── LM.txt
	  │	         └── NN.txt
	  └── EvaluationResults.txt

```

