# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:16:00 2019

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

######################### query expansion using pseudo relevance feedback algorithm
# here, we are using single term index

def topdoc_list(retrieved_):
    ''' given the retrieved top docs with tuples, return a list of top doc DOCNOS
    '''
    res = []
    for item in retrieved_:
        DOCNO,score = item
        res.append(DOCNO)
    return res

#################### TODO: should we sort the top_docs first????
def get_doc_length(collections,index_,top_docs):
    '''
    given top docs, return the term list of each docs 
    need to be adjusted to make it run faster; index compressing
    '''
    doc_dict = {}
    indexer = Indexer('stops.txt')
    for file in collections:    
        coll = open(file)
        doc = ""
        i = 0
        for line in coll:
            if '</DOC>' not in line:
                doc+=' '
                doc+=line
            else:
                # reach the end of one document 
                # pass this document 
                mydoc = doc
                doc = ""
                d = indexer.strip_text(mydoc)
                if d['DOCNO'] in top_docs:
                    
                    doc_dict[d['DOCNO']] = {}
                    if index_ == 'single':
                        indexer.single_term_index(mydoc)
                        term_list = indexer.single_term
                    if index_ == 'stem':
                        indexer.stem_index(mydoc)
                        term_list = indexer.stemmed
                    if index_ == 'positional':
                        indexer.positional_index(mydoc)
                        term_list = indexer.term_positions
                    if index_ == 'phrase':
                        indexer.phrase_index(mydoc)
                        term_list = indexer.phrases
                    for tpl in term_list:
                        lexi,info = tpl
                        if index_ == 'positional':
                            doc_dict[d['DOCNO']][lexi] = info['pos']
                        else:
                            doc_dict[d['DOCNO']][lexi] = info['tf']
                
    return doc_dict

class Expander:
    '''for both expansion and reduction 
    '''
    def __init__(self,lexicons,stop_word_path,index_):
        '''lexicons: specify the lexicon list you are gonna use
        '''
        self.stop_word_path = stop_word_path
        self.stop_words = self._set_of_stopwords(stop_word_path)
        self.doc_len = load_doc_length(index_)
        #self.doc_term = self.get_doc_length(collections,index_,)
        #self.N = len(self.doc_len)
        self.lexicons = lexicons
        self.index = index_
        #self.top_docs = []
        self.ranker = self.set_ranker()
        self.collections = os.listdir('./trec_files/')
    
    def set_ranker(self):
        #doc_len = load_doc_length(self.index)
        ranker = Ranker(self.lexicons,self.doc_len,self.stop_word_path)
        return ranker
        
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
        
    def sorting_weight(self,doc_term,term,criteria):
        '''function to compute n*idf, which is our sorting criteria
        TODO: implement several sorting criteria to see the different effects
        '''
        # compute idf
        #from query import load_doc_length,Ranker
        #doc_len = load_doc_length(self.index)
        #ranker = Ranker(self.lexicons,self.doc_len,self.stop_word_path)
        # load doc_length file 
        # top_docs = 
        #doc_term = get_doc_length(collections,self.index,)
        
        if term not in self.lexicons:
            return 0
        else:
            N = len(self.doc_len)
            pl = self.ranker.get_postinglist(term,self.index)
            df = len(pl)
            # compute idf
            idf = math.log10(N/df)
            
            df = 0
            n = 0
            f = 0
            for DOCNO,all_terms in doc_term.items():
                if term in doc_term[DOCNO]:
                    # compute number of docs in relevant set that haves term
                    f += doc_term[DOCNO][term]
                    n += 1
                
            # compute idf
            if criteria == 'number':
                return n*idf
            if criteria == 'frequency':
                return f*idf

    def topterm_list(self,q,t,criteria,df_threshold,retrieved_):
        '''
        for a given q, and a given relevant doc set,
        we want to get the top terms from the top docs
        df_threshold will prevent adding terms that are too rare in some cases 
        '''
        all_terms = {}
        res = []
        # create top doc list 
        top_docs = topdoc_list(retrieved_)
        doc_term = get_doc_length(self.collections,self.index,top_docs)
        query_terms = preprocess_query(q,self.index)
        # we want to exclude the terms that are already in the query
        for docno,term_dict in doc_term.items():
            for term,tf in term_dict.items():
                if term not in all_terms and term not in query_terms:
                    all_terms[term] = [self.sorting_weight(doc_term,term,criteria),len(self.ranker.get_postinglist(term,self.index))]
          
        sorted_tw = sorted(all_terms.items(),key=lambda kv: kv[1][0],reverse = True)
        #top_terms_weight = sorted(all_terms.items(),key=lambda kv: kv[1],reverse = True)[:t]
        for term,w_df in sorted_tw:
            if w_df[1]>df_threshold:
                res.append({term:w_df})
        if len(res)>=t:
            return res[:t]
        else:
            return res
    
    def expand_query(self,q,t,criteria,df_threshold,retrieved):
        '''
        for a given query and based on a relevant doc set, we want to expand it
        then use the new query q' to do retrieving again
        '''
        #res = []
        q_terms = preprocess_query(q,self.index)
        #retrieved = retrieve_docs(self.lexicons,self.doc_len,q,n,self.index,metric)
        added_terms = self.topterm_list(q,t,criteria,df_threshold,retrieved)
        # add terms to q_terms
        for ele in added_terms:
            for key, value in ele.items():
                q_terms.append(key)
        new_q = ' '.join(q_terms)
        return new_q
    
class Reducer:
    def __init__(self,lexicons,stop_word_path,index_):
        self.lexicons = lexicons
        self.stop_word_path = stop_word_path
        self.index = index_
        self.doc_len = load_doc_length(index_)
        self.ranker = self.set_ranker()
        
    def set_ranker(self):
        #doc_len = load_doc_length(self.index)
        ranker = Ranker(self.lexicons,self.doc_len,self.stop_word_path)
        return ranker
        
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
    
    def common_reducer(self,q,threshold):
        '''given a unpreprocessed query, we want to get the reduced query based on
        a given threshold;
        will return a new query
        note that, it is not relying on the irrelevant set and relevant set 
        '''
        res = []
        narr_ = preprocess_query(q,self.index)
        thre_dict = {}
        for term in narr_:
            if term not in thre_dict:
                thre_dict[term] = self.ranker.compute_weight(term,narr_,self.index)
        # sort the dict
        sorted_thre = sorted(thre_dict.items(),key = lambda kv:kv[1],reverse = True)
        # get the amount of the terms we are going to use here 
        term_amount = int(threshold*len(set(narr_)))
        temp = sorted_thre[:term_amount]
        for t_w in temp:
            term,weight = t_w
            res.append(term)
            
        return ' '.join(res)
        
    def rochhio_reducer(self,q,threshold,retrieved_):
        '''given a query(in our case, use the narratives), and given a relevant set(retrieved)
        reduce the query based on some threshold set for the query length
        '''
        # extract documents using original narrative
        
        #retrieved_ = retrieve_docs(lexicons,single_len,q,index_,'BM25')
        narr_terms = preprocess_query(q,self.index)
        top_docs_narr = topdoc_list(retrieved_)
        irrelevant_docs = []
        for term in narr_terms:
            
            pl = self.ranker.get_postinglist(term,self.index)
            for docno,tf in pl.items():
                if docno not in top_docs_narr:
                    # compute the term 'tf*idf'
                    irrelevant_docs.append(docno)
        # compute the term 'tf*idf'
        irre_size = len(irrelevant_docs)
        term_irre = {}
        for term in list(set(narr_terms)):
            n_irre = 0
            tf = 0
            pl = self.ranker.get_postinglist(term,self.index)
            for docno,t_f in pl.items():
                if docno in irrelevant_docs:
                    tf += t_f
                    n_irre += 1
            
            term_irre[term] = n_irre*math.log10(irre_size/len(pl))
        # sort the term_irre
        sorted_irre = sorted(term_irre.items(),key = lambda kv:kv[1], reverse = False)
        # based on threshold , removing terms that got high weight in irrelevant doc set
        all_terms = []
        for item in sorted_irre:
            term,weight = item
            all_terms.append(term)
        term_amount = int(len(set(narr_terms))*threshold)
        res = all_terms[:term_amount]
        return ' '.join(res)
        


