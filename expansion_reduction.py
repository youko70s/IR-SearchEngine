# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:19:28 2019

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


def feedback_search(raw_results,new_results,reduce_method,n,t,df_threshold,threshold):
    n = int(n)
    t = int(t)
    df_threshold = int(df_threshold)
    threshold = float(threshold)
    
    if not os.path.exists(os.path.dirname(new_results)):
        os.mkdir(os.path.dirname(new_results))
        
    narr_dict = get_narr_dict(query_file_path)
    
    reducer = Reducer(lexicons,'./stops.txt',index_)
    expander = Expander(lexicons,'./stops.txt',index_)
    print('Start using both techniques to manipulate original query(narrative).'+'\n')
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
    f.close()
    
    if reduce_method == 'common':
        start = time.time()
        for q_num,query in narr_dict.items():
            try:
            # reduce query
                new_q = reducer.common_reducer(query,threshold)
                new_q = expander.expand_query(new_q,t,'number',df_threshold,retrieved[q_num])
          
                #print(new_q)
                with open(new_results,'a') as f:
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
        print('Finished expansion+reduction using common reducer. Run time: '+str(run_time)+'\n')
    
    if reduce_method == 'rocchio':
        start = time.time()
        for q_num,query in narr_dict.items():
            try:
            # reduce query
                new_q = reducer.rocchio_reducer(query,threshold,retrieved[q_num])
                new_q = expander.expand_query(new_q,t,'number',df_threshold,retrieved[q_num])
          
                #print(new_q)
                with open(new_results,'a') as f:
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
        print('Finished expansion+reduction using rocchio reducer. Run time: '+str(run_time)+'\n')
    
    
if __name__ == "__main__":
    
    raw_results = sys.argv[1]
    new_results = sys.argv[2]
    reduce_method = sys.argv[3]
    n = sys.argv[4]
    t = sys.argv[5]
    df_threshold = sys.argv[6]
    threshold = sys.argv[7]
    #threshold = sys.argv[8]
    
    feedback_search(raw_results,new_results,reduce_method,n,t,df_threshold,threshold)

    
        
    