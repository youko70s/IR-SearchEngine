# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 02:20:53 2019

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

doc_single = load_doc_length('single')
doc_stem = load_doc_length('stem')
lexicons_single = get_inverted_index('./idf_pickle/','single')
lexicons_stem = get_inverted_index('./idf_pickle/','stem')

query_file_path = './queryfile.txt'
# to avoid too many parameters, here we are using default parameters 

metric = 'BM25'
index_ = 'single'

lexicons = lexicons_single
doc_len = doc_single


def get_narr_dict(query_file_path):
    '''function to get all the narratives
    '''
    queryfile = open(query_file_path)
    k = queryfile.read()
    q_list = []
    q_nums = []
    query = re.findall(r"<narr>\sNarrative:([\s\S]*?)</top>",k)
    querynums = re.findall(r"<num>\sNumber:([\s\S]*?)<title>",k)
    # remove all whitespace and \n
    for item in query:
        q_list.append(" ".join(item.split()))
    for no in querynums:
        q_nums.append(" ".join(no.split()))
    q_dict = dict(zip(q_nums,q_list))
    return q_dict


def reduce_search(raw_results,reduced_results,reduce_method,n,threshold):
    '''reduction method: common / rocchio
    '''
    n = int(n)
    threshold = float(threshold)
    print('Reduction on query (narrative).'+'\n')
    print('Reduction based on '+str(reduce_method)+', with threshold: '\
          +str(threshold)+'.'+'\n')
   
    narr_dict = get_narr_dict(query_file_path)
    
    if not os.path.exists(os.path.dirname(reduced_results)):
        os.mkdir(os.path.dirname(reduced_results))
        
    # use original narrative for query extraction
    
    # reduce query and do it again
    print('Start reducing query(narrative) and retrieving relevant docs'+'\n')
    reducer = Reducer(lexicons,'./stops.txt',index_)
    
    if reduce_method == 'common':
        start = time.time()
        for q_num,query in narr_dict.items():
            # reduce query
            new_q = reducer.common_reducer(query,threshold)
      
            #print(new_q)
            with open(reduced_results,'a') as f:
                docs = retrieve_docs(lexicons,doc_len,new_q,100,index_,metric)
                ranking = 0
                for file in docs:
                    docno,score = file
                    f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+metric)
                    f.write('\n')
                    ranking+=1
            f.close()
        run_time = time.time()-start
        print('Finished reducing query(narrative) using common reducer. Run time: '+str(run_time)+'\n')
    
    if reduce_method == 'rocchio':
        start = time.time()
        # we need to use raw results for rocchio reducer
        with open(raw_results) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        retrieved = {}
        for item in content:
            if item[:3] not in retrieved:
                docno = item[:3]
                retrieved[docno] = [item[5:23]]
            else:
                docno = item[:3]
                retrieved[docno].append(item[5:23])
        # only select the top n docs
        for docno,doclist in retrieved.items():
            if n>len(doclist):
                retrieved[docno] = doclist
            else:
                retrieved[docno] = doclist[:n]   
        for q_num,query in narr_dict.items():
            try:
            # reduce query
                new_q = reducer.rocchio_reducer(query,threshold,retrieved[q_num])
                #print(new_q)
                with open(reduced_results,'a') as f:
                    docs = retrieve_docs(lexicons,doc_len,new_q,100,index_,metric)
                    ranking = 0
                    for file in docs:
                        docno,score = file
                        f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+metric)
                        f.write('\n')
                        ranking+=1
                f.close()
            except: 
                pass
        run_time = time.time() - start
        print('Finished reducing query(narrative) using rocchio reducer. Run time: '+str(run_time)+'\n')

if __name__ == "__main__":
    
    #query_file_path = sys.argv[1]
    raw_results = sys.argv[1]
    reduced_results = sys.argv[2]
    #index_ = sys.argv[4]
    #metric = sys.argv[5]
    reduce_method = sys.argv[3]
    n = sys.argv[4]
    threshold = sys.argv[5]
    
    reduce_search(raw_results,reduced_results,reduce_method,n,threshold)



    