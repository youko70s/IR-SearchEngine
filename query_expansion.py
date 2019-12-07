# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:41:44 2019

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

############ main function part ######################
###### query expansion and store all results 
###### 1. use different n and t
###### 2. use different metrics
###### 3. report both run time and trec results
###### 4. use different threshold for minimum document frequency
#global doc_single,doc_stem,lexicons_single,lexicons_stem,query_file_path
doc_single = load_doc_length('single')

doc_stem = load_doc_length('stem')
lexicons_single = get_inverted_index('./idf_pickle/','single')
lexicons_stem = get_inverted_index('./idf_pickle/','stem')
query_file_path = './queryfile.txt'
# to avoid too many parameters, here we are using default parameters 
#global metric,index_,lexicons,doc_len

metric = 'BM25'
index_ = 'single'
lexicons = lexicons_single
doc_len = doc_single

############TODO: add this function to query.py ###########
def get_query_dict(query_file_path):
    queryfile = open(query_file_path)
    k = queryfile.read()
    q_list = []
    q_nums = []
    query = re.findall(r"<title>\sTopic:([\s\S]*?)<desc>",k)
    querynums = re.findall(r"<num>\sNumber:([\s\S]*?)<title>",k)
    # remove all whitespace and \n
    for item in query:
        q_list.append(" ".join(item.split()))
    for no in querynums:
        q_nums.append(" ".join(no.split()))
    q_dict = dict(zip(q_nums,q_list))
    return q_dict

def expand_search(raw_results_path,expanded_results,n,t,threshold):
    n = int(n)
    t = int(t)
    threshold = int(threshold)
    print('Starting expanding query and retrieving relevant documents')
    print('Expansion based on top '+str(n)+' docs, top '+str(t)+' terms, with df threshold '\
          +str(threshold)+'.'+'\n')
    start = time.time()

    q_dict = get_query_dict(query_file_path)
    if not os.path.exists(os.path.dirname(expanded_results)):
        os.mkdir(os.path.dirname(expanded_results))
    with open(raw_results_path) as f:
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
        if n > len(doclist):
            retrieved[docno] = doclist
        else:
            retrieved[docno] = doclist[:n]   
    
    expander = Expander(lexicons,'./stops.txt',index_)
    for q_num,query in q_dict.items():
        try:
            # expand query
            new_q = expander.expand_query(query,t,'number',threshold,retrieved[q_num])
            #print(new_q)
            with open(expanded_results,'a') as f:
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
        
    run_time = time.time()-start
    print('Finished query expansion and retieved relevant docs.'+'\n')
    print('Run time: '+str(run_time)+'\n')
    

if __name__ == "__main__":
    
    #query_file_path = sys.argv[1]
    raw_results_path = sys.argv[1]
    expanded_results = sys.argv[2]
    #index_ = sys.argv[4]
    #metric = sys.argv[5]
    n = sys.argv[3]
    t = sys.argv[4]
    threshold = sys.argv[5]
    
    expand_search(raw_results_path,expanded_results,n,t,threshold)


