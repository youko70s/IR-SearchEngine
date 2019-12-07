# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:17:57 2019

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
    thre_set = [0.7,0.5,0.2]
    for thre in thre_set:
        result_file = result_dir + 'common_'+str(thre)[2]+'.txt'
        reduce_search('./baseline/narrative.txt',result_file,'common',5,thre)
        result_file = result_dir +'rocchio_'+str(thre)[2]+'.txt'
        reduce_search('./baseline/narrative.txt',result_file,'rocchio',5,thre)
    print('Reduction demo done!'+'\n')
if __name__ == "__main__":
    
    #query_file_path = sys.argv[1]
    result_dir = sys.argv[1]
    
    main(result_dir)

 