# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:46:46 2019

@author: youko
"""
import os
import sys
import re
import time 
import pickle
from nltk.stem import PorterStemmer 
import math
import json
from build import Indexer,Parser
from query import Ranker
from query import preprocess_query, get_inverted_index,retrieve_docs,load_doc_length
#################################################

def proximity_retrieval(lexicons_pos,doc_pos,query,top_N,k):
    ''' using proximity index to retrieve documents
    '''
    Q = preprocess_query(query,'positional_index')
    ranker = Ranker(lexicons_pos,doc_pos,'stops.txt')
    
    ranking_docs = {}
    for i in range(len(Q)-1):
        try:
            match,matched_docs = ranker.proximity_compare(Q[i],Q[i+1],k)
        
            for docno,matched in matched_docs.items():
                if docno not in ranking_docs:
                    ranking_docs[docno] = len(matched)
                else:
                    
                    ranking_docs[docno] += len(matched)
        except:
            pass
    res = sorted(ranking_docs.items(),key = lambda x:x[1],reverse = True)
    if len(res)>top_N:
        res = res[:top_N]
    #else:
        # pass to single_term using lm model
        #new_docdict = get_doc_length(files,'single_term')
        #res = retrieve_docs(new_docdict,query,top_N,'single_term_index','LM')
    return res
    
def phrase_search(query,lexicons_phrases,doc_phrases):
 
    Q = preprocess_query(query,'phrases')
    freq = []
    ranker = Ranker(lexicons_phrases,doc_phrases,'stops.txt')

    for phrase in Q:
        pl = ranker.get_postinglist(phrase,'phrases')
        try:
            freq.append(len(pl))
        except:
            freq.append(0)
    if sum(freq)>=5:
        # the phrase is frequent enough to use 
        return True
    else:
        return False


def main(index_path,query_file_path,results_file):
    '''main function for using proximity and phrase index for document retrieval
    '''
    print('Start dynamic retrieving relavant documents...'+'\n')
    queryfile = open(query_file_path)
    if not os.path.exists(os.path.dirname(results_file)):
        os.mkdir(os.path.dirname(results_file))
        
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
    doc_pos = load_doc_length('positional')
    doc_phrases = load_doc_length('phrase')
    doc_single = load_doc_length('single')
    
    lexicons_single = get_inverted_index(index_path,'single_term_index')
    lexicons_phrases = get_inverted_index(index_path,'phrases')
    lexicons_pos = get_inverted_index(index_path,'positional_index')
    start = time.time()
    for q_num,query in q_dict.items():
        if phrase_search(query,lexicons_phrases,doc_phrases):
            # using phrase index to retrieve documents
            retrieved = retrieve_docs(lexicons_phrases,doc_phrases,query,100,'phrases','LM')
            with open(results_file,'a') as f:
                #for q_num,query in q_dict.items():
                #retrieved = retrieve_docs(doc_phrases,query,100,'phrases','LM')
                #with open(score_res,'w') as f:
                ranking = 0
                for file in retrieved:
                    docno,score = file
                    f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+'PHRASES')
                    f.write('\n')
                    ranking+=1
       
                # if not enough docs are retrieved:
                if len(retrieved)<=100:
                    # pass to proximity search
                    pos_retrieved = proximity_retrieval(lexicons_pos,doc_pos,query,100,3)
                    new_pos_retrieved = []
                    for file in pos_retrieved:
                        docno,score = file
                        if docno not in retrieved:
                            new_pos_retrieved.append((docno,score))
                            f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+'PROXIMITY')
                            f.write('\n')
                            ranking +=1
                    if len(retrieved)+len(pos_retrieved)<=100:
                        # pass to single term index
                        
                        single_retrieved = retrieve_docs(lexicons_single,doc_single,query,100,'single_term_index','LM')
                        for file in single_retrieved:
                            docno,score = file
                            if docno not in retrieved and docno not in new_pos_retrieved:
                                #new_pos_retrieved.append((docno,score))
                                f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+'SINGLE_LM')
                                f.write('\n')
                                ranking +=1
        else:
            with open(results_file,'a') as f:
                pos_retrieved = proximity_retrieval(lexicons_pos,doc_pos,query,100,3)
                #new_pos_retrieved = []
                ranking = 0
                for file in pos_retrieved:
                    docno,score = file
                    
                    #new_pos_retrieved.append((docno,score))
                    f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+'PROXIMITY')
                    f.write('\n')
                    ranking +=1
                if len(pos_retrieved)<=100:
                    single_retrieved = retrieve_docs(lexicons_single,doc_single,query,100,'single_term_index','LM')
                    for file in single_retrieved:
                        docno,score = file
                        if docno not in pos_retrieved:
                            #new_pos_retrieved.append((docno,score))
                            f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+'SINGLE_LM')
                            f.write('\n')
                            ranking +=1
        f.close()
    time_1 = time.time()-start
    print('Finished retrieving documents.'+'\n'+\
          'Running time for query dynamic:'+str(time_1)+'\n')


                    
                
if __name__ == "__main__":
    index_path = sys.argv[1]
    query_file_path = sys.argv[2]
    results_file = sys.argv[3]
    main(index_path,query_file_path,results_file)


