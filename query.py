# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:40:46 2019

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
from build import Indexer,Parser
#################################################

# preprocess queries with the same rules as indexing

def preprocess_query(query,index_):
    ''' this function will take a query form and then do preprocessing
    it will accept a query and then return lowercased item list
    parameter index_: 'stem_index' or 'single_term_index' or 'phrases'
    '''
    # lowercase everything 
    query = query.lower()
    # import indexer to do preprocessing
    tokenizer = Parser('stops.txt')
    stop_words = tokenizer.stop_words

    terms = tokenizer.parse(query)
    res = []
    for item in terms:
        if type(item) == list:
            for i in item:
                res.append(i)
        else:
            res.append(item)
    if index_ == 'single_term_index':
        new_res = [term for term in res if term not in stop_words]
    if index_ == 'stem_index':
        ps = PorterStemmer()
        new_res = [ps.stem(term) for term in res if term not in stop_words]
    if index_ == 'positional_index':
        new_res = res
    if index_ == 'phrases':
        tokenizer.generate_phrases(query)
        new_res = tokenizer.two_grams+tokenizer.three_grams
    return new_res


# import all indexes
def get_inverted_index(index_path,index_):
    '''this function will take a memory constraint value, and index name, 
    and will return the dict of inverted index we get at first stage
    '''
    all_indexes = os.listdir(index_path)
    for index_file in all_indexes:
        if index_ in index_file:
            tgt_file = index_file
            break
    filename = index_path + tgt_file
    file = open(filename,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def load_doc_length(index_):
    '''this function will read in the doc length backup file we created
    at first building stage
    '''
    path = './document_length_dict/'
    all_dicts = os.listdir(path)
    for file in all_dicts:
        if index_ in file:
            tgt_file = file
            break
        
    filename = path + tgt_file
    f = open(filename,'rb')
    object_file = pickle.load(f)
    f.close()
    return object_file

class Ranker:
    def __init__(self,lexicons,doc_dict,stop_word_path):
        self.lexicons = lexicons
        self.stop_words = self._set_of_stopwords(stop_word_path)
        #self.cache = cache
        self.N = len(doc_dict)
        self.docdict = doc_dict
        self.avgdl = self.compute_avgdl()
        self.cl = self.compute_cl()
        ####
        #self.indexer = Indexer()
    def _set_of_stopwords(self, path):
        '''
        create set of words from stop-words file
        :param path: path to stop-word file
        :return: set of stop-words
        '''
        if not os.path.exists(path):
            return set()
        with open(path) as f:
            return set((line.strip() for line in f.readlines()))
    
    def transform_pl(self,raw_pl,index_):
        '''given a raw posting list in a list form, we want to transfer it to a dict
        '''
        new_pl = {}
        for item in raw_pl:
            json_str = item.replace("'", "\"")
            d = json.loads(json_str)
            if index_ == 'positional_index':
                new_pl[d['DOCNO']] = d['pos']
            else:
                new_pl[d['DOCNO']] = d['tf']
        return new_pl

    def compute_avgdl(self):
        try:
            total = 0
            for key,value in self.docdict.items():
                #for subkey,subv in value.items():
                total += value
            return total/self.N
        except:
            return 0
    
    def compute_cl(self):
        try:
            cl = 0
            for key,value in self.docdict.items():
                cl += self.get_docsize(key)
            return cl
        except:
            return 0
    
    def get_docsize(self,DOCNO):
        try:
            #size = 0
            return self.docdict[DOCNO]
        except:
            return 0
            
            
    def get_postinglist(self,term,index_):
        '''given a term, get its posting list
        '''
        try:
            raw_pl = self.lexicons[term]
            pl = self.transform_pl(raw_pl,index_)
            return pl
        except:
            pass
        
    ## implementation of VSM model
    def compute_tfidf(self,term,DOCNO,index_):
        ''' function to compute tfidf
        for one specific term with a specific document
        '''
        if term not in self.lexicons:
            return 0
        else:
            N = self.N
            pl = self.get_postinglist(term,index_)
            if DOCNO not in pl:
                return 0
            else:
                tf = pl[DOCNO]
                df = len(pl)
                # compute idf
                idf = math.log10(N/df)
                return tf*idf

    
    def compute_weight(self,term,query,index_):
        if term not in self.lexicons:
            return 0
        else:
            N = self.N
            pl = self.get_postinglist(term,index_)
            tf = query.count(term)
            df = len(pl)
            idf = math.log10(N/df)
            return tf*idf
    
        
    def cosine_similarity(self,query,DOCNO,index_):
        '''take a preprocessedquery and a document, 
        compute and return the cossine similarity
        '''
        try:
            dot_product = 0
            d_under = 0
            w_under = 0
            set_terms = list(set(query))
            for term in set_terms:
                dot_product += self.compute_tfidf(term,DOCNO,index_)*self.compute_weight(term,query,index_)
                d_under += self.compute_tfidf(term,DOCNO,index_)**2
                w_under += self.compute_weight(term,query,index_)**2
            return dot_product/(math.sqrt(d_under+w_under))
        except ZeroDivisionError:
            return 0
     
    
    ## implementation of BM25 model
    def compute_bm25_sc(self,term,DOCNO,query,index_):
        ''' function to compute a term and its sc with a doc in a specific query
        '''
        # parameters for bm25 model
        k1 = 1.2
        k2 = 2
        b = 0.75
        if term not in self.lexicons:
            return 0
        else:
            try:
                pl = self.get_postinglist(term,index_)
                # weight = idf
                N = self.N
                n = len(pl)
                w = math.log10((N-n+0.5)/n+0.5)
                tf = pl[DOCNO]
                qtf = query.count(term)
                avgdl = self.avgdl
                d_size = self.get_docsize(DOCNO)
                
                return w*((k1+1)*tf/(tf+k1*(1-b+b*d_size/avgdl)))*((k2+1)*qtf/(k2+qtf))
            except:
                return 0
            
    def bm25_similarity(self,query,DOCNO,index_):
        '''function to compute bm25 similarity with a query to a document
        '''
        try:
            sc = 0
            set_terms = list(set(query))
            for term in set_terms:
                sc += self.compute_bm25_sc(term,DOCNO,query,index_)
            return sc
        except:
            return 0
    
    def compute_lm_sc(self,term,DOCNO,query,index_):
        '''compute a single term's similarity to a document using lm
        given query
        '''
        mu = self.avgdl
        if term not in self.lexicons:
            return 0
            
        else:
            pl = self.get_postinglist(term,index_)
            if DOCNO not in pl:
                dtf = 0
            else:
                dtf = pl[DOCNO]
            ctf = 0
            for key,value in pl.items():
                ctf+=value
            cl = self.cl
            dl = self.get_docsize(DOCNO)
        
            return math.log10((dtf+mu*ctf/cl)/(dl+mu))
            
    
    def lm_similarity(self,query,DOCNO,index_):
        '''function to compute lm similarity given a query
        '''
        try:
            sc = 0
            set_terms = list(set(query))
            for term in set_terms:
                sc += self.compute_lm_sc(term,DOCNO,query,index_)
            return sc
        except:
            return 0
        
    def proximity_compare(self,t1,t2,k):
        '''get proximitity comparation between 2 terms
        '''
       
        pl_1 = self.get_postinglist(t1,'positional_index')
        pl_2 = self.get_postinglist(t2,'positional_index')
        
        match = 0
        match_docs = {}
        # search from the less frequent term
        if len(pl_1)<=len(pl_2):
            # search from pl_1
            for docno,pos in pl_1.items():
                if docno in pl_2:
                    pos_2 = pl_2[docno]
                    if len(pos)<=len(pos_2):
                        for pos1 in pos:
                            for pos2 in pos_2:
                                # parameter k = 5 in our case
                                if abs(pos2-pos1)<=k:
                                    match+=1
                                    if docno not in match_docs:
                                        match_docs[docno] = [(pos1,pos2)]
                                    else:
                                        match_docs[docno].append((pos1,pos2))
                                    break
                    else:
                        for pos2 in pos_2:
                            for pos1 in pos:
                                # parameter k = 5 in our case
                                if abs(pos2-pos1)<=5:
                                    match+=1
                                    if docno not in match_docs:
                                        match_docs[docno] = [(pos1,pos2)]
                                    else:
                                        match_docs[docno].append((pos1,pos2))
                                    break
        else:
            for docno,pos in pl_2.items():
                if docno in pl_1:
                    pos_1 = pl_1[docno]
                    if len(pos)<=len(pos_1):
                        for pos2 in pos:
                            for pos1 in pos_1:
                                # parameter k = 5 in our case
                                if abs(pos2-pos1)<=5:
                                    match+=1
                                    if docno not in match_docs:
                                        match_docs[docno] = [(pos1,pos2)]
                                    else:
                                        match_docs[docno].append((pos1,pos2))
                                    break
                    else:
                        for pos1 in pos_1:
                            for pos2 in pos:
                                # parameter k = 5 in our case
                                if abs(pos2-pos1)<=5:
                                    match+=1
                                    if docno not in match_docs:
                                        match_docs[docno] = [(pos1,pos2)]
                                    else:
                                        match_docs[docno].append((pos1,pos2))
                                    break
        return match,match_docs


def retrieve_docs(lexicons,doc_dict,query,top_N,index_,metric):
    '''given a query, retrieve the documents from the collection
    write the results to file
    index_: single_term_index or stem_index or phrase index 
    '''
    Q = preprocess_query(query,index_)
    #lexicons = get_inverted_index(path)
    
    #doc_dict = get_doc_length(files,index_)
    #if index_ == 'single_term_index':
    ranker = Ranker(lexicons,doc_dict,'stops.txt')

    possible_docs = []
    score = {}
    for term in Q:
        try:
            pl = ranker.get_postinglist(term,index_)
            for key,value in pl.items():
                possible_docs.append(key)
        except:
            pass
    for doc in possible_docs:
        if metric == 'COSINE':
            score[doc] = ranker.cosine_similarity(Q,doc,index_)
        if metric == 'BM25':
            score[doc] = ranker.bm25_similarity(Q,doc,index_)
        if metric == 'LM':
            score[doc] = ranker.lm_similarity(Q,doc,index_)
            
    res = sorted(score.items(),key = lambda x:x[1],reverse = True)
    if len(res)>top_N:
        res = res[:top_N]
    return res
        

def main(index_path,query_file_path,metric,index_,results_file):
    ''' main function for processing the query and get the retrieved documents
    '''
    
    metric = metric.upper()
    print('Start retrieving relavant documents using '+metric+' '+'as metric with index: '+index_+'...'+'\n')
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


    doc_single = load_doc_length('single')
    doc_stem = load_doc_length('stem')
    
    lexicons_single = get_inverted_index(index_path,'single_term_index')

    lexicons_stem = get_inverted_index(index_path,'stem_index')
    if index_ =='single':
        lexicons = lexicons_single
        doc_dict = doc_single
        indexing = 'single_term_index'
    if index_ == 'stem':
        lexicons = lexicons_stem
        doc_dict = doc_stem
        indexing = 'stem_index'

    start = time.time()
    #doc_dict = get_doc_length(files,'single_term_index')
    for q_num,query in q_dict.items():
        with open(results_file,'a') as f:
        #for q_num,query in q_dict.items():
            retrieved = retrieve_docs(lexicons,doc_dict,query,100,indexing,metric)
            #with open(score_res,'w') as f:
            ranking = 0
            for file in retrieved:
                docno,score = file
                f.write(q_num+' '+'0'+docno+str(ranking)+' '+str(score)+' '+metric)
                f.write('\n')
                ranking+=1
        f.close()
    time_1 = time.time()-start
    print('Finished retrieving documents.'+'\n')
    print('Running time:'+str(time_1)+'\n')

            
if __name__ == "__main__":
    index_path = sys.argv[1]
    query_file_path = sys.argv[2]
    metric = sys.argv[3]
    index_ = sys.argv[4]
    results_file = sys.argv[5]
    main(index_path,query_file_path,metric,index_,results_file)



