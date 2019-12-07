# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:30:05 2019

@author: youko
"""

import os
import sys
import re
import datetime
import time 
import pickle
from nltk.stem import PorterStemmer 
import math
import json
from build import *
from query import *
from relevance_feedback import *
from query_reduction import *
from query_expansion import *
from expansion_reduction import *

doc_single = load_doc_length('single')
doc_stem = load_doc_length('stem')
lexicons_single = get_inverted_index('./idf_pickle/','single')
lexicons_stem = get_inverted_index('./idf_pickle/','stem')

query_file_path = './queryfile.txt'

metric = 'BM25'
index_ = 'single'

lexicons = lexicons_single
doc_len = doc_single

def main(result_file):
    
    feedback_search('./baseline/narrative.txt',result_file,'rocchio',5,10,2,0.5)

if __name__ == "__main__":
    
    #query_file_path = sys.argv[1]
    result_file = sys.argv[1]
    
    main(result_file)

 
    
    
    
    