# IR-SearchEngine
This repository provides a classical implementation of information retieval search engine buillt from scratch. 

## How to run 
In this repo, I already provided some sample trec documents in [trec_files](https://github.com/youko70s/IR-SearchEngine/tree/master/trec_files). (See information about trec: [TREC](https://trec.nist.gov/)); I also provided a sample queryfile for retrieving relavant documents.

Copy paste the following command line to clone this repository to your local:

    git clone https://github.com/youko70s/IR-SearchEngine.git

Once being cloned, go to your local directory `IR-SearchEngine`.

### building the index 

    python build.py [trec-files-dir-path] [index-type] [output-dir]

Builds the index, accepts 3 arguments

* `[trec-files-dir-path]` the directory containing the raw documents (e.g. fr940104.0). (In this repo, the raw documents are located in `./trec_files/`)
* `[index-type]` can be one of the following: 'single', 'stem', 'positional', 'phrase'.
* `[output-dir]` the directory where your index and lexicon files will be written (if not existed, then the program will create a new directory)

example command for building the index:

    python build.py ./trec_files/ stem ./my_indexes/



