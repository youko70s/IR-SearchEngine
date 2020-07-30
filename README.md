# IR-SearchEngine
This repository provides a classical implementation of information retrieval search engine buillt from scratch. 

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

In this experiment, I implemented pseudo relevance feedback to do query expansion. 

    $ python query_expansion.py [raw-results-path] [expanded-results-path] [n] [t] [threshold]

* `[raw-results-path]` path of the retrieved results using raw queries. (Here: `./baseline/title.txt`)
* `[expanded-results-path]` path to store the retrieved results using expanded query
* `[n]` number of top documents (i.e. number of documents in relevant set)
* `[t]` number of top terms in relevant set
* `[threshold]` minimum document frequency for the terms to be added 

example command:

    $ python query_expansion.py ./baseline/title.txt ./relevance_feedback/expansion.txt 10 10 2

In pseudo feedback system, there are several issues we want to consider. How many top documents are we going to use? How many top terms are we going to pick? Does the added terms make sense?

To measure these parameters and how they affect the quality of the engine, I tried different `n`, `t`, `df_threshold`. When measuring one parameter, I fixed the other two variables.

- `n`: [50,20,5,1]
- `t`: [20,10,5]
- `df_threshold`: [10,5,2]

To make it easier, I created a demo file for trying all these parameters. To run it:

    $ python expansion_demo.py [results_dir]

* `results_dir`: specify the directory you want to store the expansion demo results for various parameters.

### Query Reduction

In this experiment, I implemented two methods for reducing a query

    $ python query_reduction.py [raw-results-path] [reduced-results-path] [reduce-method] [n] [threshold]

* `[raw-results-path]` path of the retrieved results using raw queries. (Here: `./baseline/narrative.txt`)
* `[expanded-results-path]` path to store the retrieved results using expanded query
* `[reduce-method]` should be one the the following: `rocchio`, `common`
* `[n]` number of top documents used for rocchio reduction (i.e. number of documents in relevant set)
* `[threshold]` reduced size proportion of the raw query. Should be a value between 0-1.

An example command would be:

    $ python query_reduction.py ./baseline/narrative.txt ./relevance_feedback/reduction.txt rocchio 10 0.5

About the two methodologies:

* index pruning with query threshold (common): this is using the similar idea as index pruning. Based on the threshold (0~1), it will select the terms with largest tf*idf and use them as the reduced query. 
* rocchio: this is based on rocchio algorithm.

To measure the performance of the two methodologies, and the effect of threshold, I let the threshold to be 0.2,0.5,0.7 for both two methodologies. (notice that I fixed n to be 5). To make it easier, I also created a demo file for you to try out. To run it:

    $ python reduction_demo.py [results_dir]

* `results_dir`: specify the directory you want to store the reduced demo results for various parameters.

### Query Expansion + Reduction

This is the implementation for using both expansion and reduction. 

    $ python [raw-results-path] [new-results-path] [reduce-method] [n] [t] [df_threshold] [threshold]

The parameters here are the same as mentioned above. 

example command:

    $ python ./baseline/narrative.txt ./relevance_feedback/both_implementation.txt common 5 10 2 0.5

I have chosen the optimal parameters in this case and created a demo file for you to run directly. To run it:

    $ python both_demo.py [results_dir]


## Report 

I conducted reports for further analysis and discussion over the results throughout the process. The reports are available upon request. If you are interested in, please email: [youko1970s@gmail.com](mailto:youko1970s@gmail.com).