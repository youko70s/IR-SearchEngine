# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:06:34 2019

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

def main(result_dir):
    ###### n part already done
    n_set = [50,20,5,1]
    for n in n_set:
        result_file = result_dir+'n_'+str(n)+'.txt'
        expand_search('./baseline/title.txt',result_file,n,10,2)
        
    t_set = [20,10,5]
    for t in t_set:
        # do query expansion
        result_file = result_dir+'t_'+str(t)+'.txt'
        expand_search('./baseline/title.txt',result_file,5,t,2)
        
    df_threshold_set = [10,5,2]
    for thre in df_threshold_set:
        result_file = result_dir+'df_threshold_'+str(thre)+'.txt'
        expand_search('./baseline/title.txt',result_file,5,10,thre)
    print('Expansion demo done!'+'\n')
    
        

if __name__ == "__main__":
    
    #query_file_path = sys.argv[1]
    result_dir = sys.argv[1]
    
    main(result_dir)
