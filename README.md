# IR-SearchEngine
This repository provides a classical implementation of information retieval search engine buillt from scratch. 

In this repo, I already provided some sample trec documents in [trec_files](https://github.com/youko70s/IR-SearchEngine/tree/master/trec_files). (See information about trec: [TREC](https://trec.nist.gov/)); I also provided a sample queryfile for retrieving relavant documents.

Copy paste the following command line to clone this repository to your local:

    $ git clone https://github.com/youko70s/IR-SearchEngine.git

Once being cloned, go to your local directory `IR-SearchEngine`.

## Installation

All libraries are standard python libraries. However, you have to install `nltk` as to import PorterStemmer. 


Run the following command in your terminal:

    $ pip install -r requirements.txt


## Building the Index 

    $ python build.py [trec-files-dir-path] [index-type] [output-dir]

Builds the index, accepts 3 arguments

* `[trec-files-dir-path]` the directory containing the raw documents (e.g. fr940104.0). (In this repo, the raw documents are located in `./trec_files/`)
* `[index-type]` can be one of the following: 'single', 'stem', 'positional', 'phrase'.
* `[output-dir]` the directory where your index and lexicon files will be written (if not existed, then the program will create a new directory)

example command for building the index:

    $ python build.py ./trec_files/ stem ./my_indexes/

## Query Processing

We provided two methods for processing queries and retrieving relavant documents. 

### Static Query


    $ python query.py [index-dir-path] [query-file-path] [metric] [index-type] [results-file]

* `[index-dir-path]` takes the path of the directory where you store your index files (note that it should align with what you specified in the building phase)
* `[query-file-path]` path to the query file
* `[metric]` retrieval model used for searching documents. Can be one of the following: 'cosine', 'bm25', 'lm'
* `[index-type]` one of the following: 'single', 'stem'
* `[results-file]` path to the results file that you are going to store the retrieval results

We provided two methods for processing queries and retrieving relavant documents. 

In static query processing, we will use either single term index or stem index for retrieving purpose. For better understanding of application mechanism and easier evaluation on the engine's performance, we will use the TREC query sample file identified by their unique numbers. In this phase, we only use `title` part of the queries for retrieving documents. 

example command for static query processing:

    $ python query.py ./my_indexes/ ./data/queryfile.txt bm25 single ./results/single_bm25.txt

### Dynamic Query

    $ python query_dynamic.py [index-dir-path] [query-file-path] [results-file]

Same as static query processing, except that in this case, we are using phrase index together with positional index for dynamically retrieving purpose based on some thresholds (I already provided default thresholds in the script files, but you can also alter these thresholds yourself). Takes 3 arguments:

* `[index-dir-path]` path to the directory where you store your index files at the building phase
* `[query-file-path]` path to the query file
* `[results-file]` path to the results file that you are going to store the retrieval results

example command for dynamic query processing:

    $ python query_dynamic.py ./my_indexes/ ./data/queryfile.txt ./results/dynamic.txt

## Relevance Feedback

In this experiment, I implemented both query expansion and reduction techniques. I implemented pseudo relevance feedback (for query expansion), rocchio relevance feedback and index pruning (in the below, referring as 'common') for query reduction. I also implement a way to use both expansion and reduction. 

To better compare those methods and how parameters could affect the performance, we set our baseline models as:

* Expansion: use `title` in `queryfile.txt`. [title.txt](https://github.com/youko70s/IR-SearchEngine/blob/master/baseline/title.txt)
* Reduction : use `narrative` in `queryfile.txt`. [narrative.txt](https://github.com/youko70s/IR-SearchEngine/blob/master/baseline/narrative.txt)
* Expansion + Reduction: use `narrative` in `queryfile.txt`. [narrative.txt](https://github.com/youko70s/IR-SearchEngine/blob/master/baseline/narrative.txt)

I have already put baseline results in [baseline](https://github.com/youko70s/IR-SearchEngine/tree/master/baseline). 

Considering performances using different index and metric done before, the baseline results were retrieved based on `single term index` using `BM25` as metric. 

To see the implementations of `Expander` and `Reducer`, check: [relevance_feedback.py](https://github.com/youko70s/IR-SearchEngine/blob/master/relevance_feedback.py)

### Query Expansion 

In this experiment, I implement pseudo relevance feedback to do query expansion. 

    $ python query_expansion.py [raw-results-path] [expanded-results-path] [n] [t] [threshold]

* `[raw-results-path]` path of the retrieved results using raw queries. (Here: `./baseline/title.txt`)
* `[expanded-results-path]` path to store the retrieved results using expanded query
* `[n]` number of top documents (i.e. number of documents in relevant set)
* `[t]` number of top terms in relevant set
* `[threshold]` minimum document frequency for the terms to be added 

example command:

    python query_expansion.py ./baseline/title.txt ./relevance_feedback/expansion.txt 10 10 2

In pseudo feedback system, there are several issues we want to consider. How many top documents are we going to use? How many top terms are we going to pick? Does the added terms make sense?


