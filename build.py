#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:10:26 2019

@author: youko
"""

import os
import sys
import re
import datetime
from datetime import datetime
import time 
import pickle
from nltk.stem import PorterStemmer 
import shutil


class Parser:
    def __init__(self, stop_word_path):
        self.contain_number = re.compile(".*\d.*")
        self.punctuations = ["|", "^", "!", "?", "*", ";",'<','>',"'", "\\", '"', '&', '(', ')', '+', '=', '÷',
                                ']', '[', '\n', '\t', '#', ' %', '`', 'כ', 'ז', 'ף', 'ר', 'ד', 'כ', '}', 'ק', 'ם']
        self.months = {'january':'01','february':'02','march':'03','april':'04',\
                  'may':'05','june':'06','july':'07','august':'08','september':'09',\
                  'october':'10','november':'11','december':'12'}
        self.normal_date = re.compile("\d+/\d+|\d+/\d+/\d+")
        self.stop_words = self._set_of_stopwords(stop_word_path)
        self.special_toks = []
        self.two_grams = []
        self.three_grams = []

    def digit_format(self,text):
        ''' for some given text, we want to process the digital formats
        '''
        result = []
        pt_1 = re.findall(r'\d+,*\d*\.[0]+',text)
        pt_2 = re.findall(r'\d+,\d+($|[\s|\.|\,|!|\"|\'|=|)|(|//|\\])',text)
        if len(pt_1)!=0 and len(pt_2)!=0:
            newtxt = re.sub(r'(\d+,*\d*)\.[0]+', r'\1', text.rstrip())
            refind_pt_2 = re.findall(r'\d+,\d+[\s|\.|\,|!|\"|\'|=|)|(|//|\\]',newtxt)
            result = re.sub(r'(\d+),(\d+)($|[\s|\.|\,|!|\"|\'|=|)|(|//|\\])',r'\1\2',newtxt.rstrip())
        elif len(pt_1)!=0:
            result = re.sub(r'(\d+,*\d*)\.[0]+', r'\1', text.rstrip())
        elif len(pt_2)!=0:
            result = re.sub(r'(\d+),(\d+)($|[\s|\.|\,|!|\"|\'|=|)|(|//|\\])',r'\1\2',text.rstrip())

        return result 
        
    def special_tokens(self,word):
        '''given some word, we want to preprocess it
        '''
        # case 1: Ph.D， U.S.A
        prefixes = ['pre','re','post','anti','semi','extra','sub','multi']
        # email
        if re.findall(r'[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',word):
            result = re.findall(r'[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',word)[0]
        # file extensions
        elif re.findall(r'([^\s]*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML|csv|CSV|TXT|txt)',word):
            result = re.sub(r'([^\s]*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML|csv|CSV|TXT|txt)',r'\1\2',word).lower()
       # IP address
        elif re.findall(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',word):
            result = re.findall(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',word)[0]
       # URL
        elif re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',word):
            result = re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',word)[0]
        # abbreviations
        elif re.findall(r'(?:[A-Za-z]+\.)+[a-zA-Z]+',word):
            z = re.findall(r'(?:[A-Za-z]+\.)+[a-zA-Z]+',word)[0]
            result = re.sub('\.','',z).lower()
        
        # date
        elif re.findall(r'(\d+)\/(\d+)\/(\d+)',word):
            result = re.findall(r'\d+\/\d+\/\d+',word)[0]
            
        elif len(self.digit_format(word))!=0:
            result = self.digit_format(word)
        
        # case 2: monetary symbols
        elif re.findall(r'\$\d*(?:\.\d*)?',word):
            result = re.findall(r'\$\d*(?:\.\d*)?',word)[0]
        # case 3: alphabet-digit
        elif re.findall(r'[a-z|A-Z]+\-[\d]+',word):
            z = re.findall(r'[a-z|A-Z]+\-[\d]+',word)[0]
            # return two things 
            res_1 = re.sub('\-','',z).lower()
            alp = re.findall(r'[a-z|A-Z]+',z)[0]
            res_2 = alp.lower()
            result = [res_1,res_2]
        # case 4: digit-alphabet
        elif re.findall(r'[\d]+\-[a-z|A-Z]{3,}',word):
            z = re.findall(r'[\d]+\-[a-z|A-Z]{3,}',word)[0]
            res_1 = re.sub('\-','',z).lower()
            alp = re.findall(r'[a-z|A-Z]+',z)[0]
            res_2 = alp.lower()
            result = [res_1,res_2]
        # case 5: prefix
        elif re.findall(r'[a-z|A-Z]+(?:-[a-z|A-Z]+)+',word):
            result = []
            z = re.findall(r'[a-z|A-Z]+(?:-[a-z|A-Z]+)+',word)[0]
            z = z.lower()
            for prefix in prefixes:
                if z.startswith(prefix):
                    # remove prefix and store
                    res_1 = re.sub('\-','',z)
                    res_2 = re.sub(prefix,'',res_1)
                    result = [res_1,res_2]
                    break
            if len(result)==0:
           # not hyphened word, we want to keep everything here
                
                res_1 = re.sub('\-',' ',z)
                result = re.split(' ',res_1)
                res_2 = re.sub('\-','',z)
                result.append(res_2)
        # z = re.findall(r'^[\$][\d\.]$',word)[0]
        
        return result
    
    def cut_special_toks(self,text):
        '''given some text, we want to preprocess it
        '''
        # case 1: Ph.D， U.S.A
        prefixes = ['pre','re','post','anti','semi','extra','sub','multi']
        # email
        if re.findall(r'[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',text):
            result = re.findall(r'[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?',text)
            for item in result:
                # substitude all the special tokens to , so that we can easily split the text
                text = re.sub(item,',',text)                       
        # file extensions
        if re.findall(r'([^\s]*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML|csv|CSV|TXT|txt)',text):
            result = re.findall(r'([^\s]*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML|csv|CSV|TXT|txt)',text)
            for i,ele in enumerate(result):
                result[i] = '.'.join(ele)
            for item in result:
                text = re.sub(item,',',text)                       
    
            #result = re.sub(r'([^\s]*)\.(jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|html|HTML|csv|CSV|TXT|txt)',r'\1\2',word).lower()
       # IP address
        if re.findall(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',text):
            result = re.findall(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',text)
            for item in result:
                text = re.sub(item,',',text) 
       # URL
        if re.findall(r'http|HTTP[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',text):
            result = re.findall(r'http|HTTP[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',text)
            for item in result:
                text = re.sub(item,',',text)
        # abbreviations
        if re.findall(r'(?:[A-Za-z]+\.)+[a-zA-Z]+',text):
            result = re.findall(r'(?:[A-Za-z]+\.)+[a-zA-Z]+',text)
            for item in result:
                text = re.sub(item,',',text)
        
        # date
        if re.findall(r'\d+\/\d+\/\d+',text):
            result = re.findall(r'\d+\/\d+\/\d+',text)
            for item in result:
                text = re.sub(item,',',text)
        
        # case 2: monetary symbols
        if re.findall(r'\$\d*[?:\.\d*]+',text):
            result = re.findall(r'\$\d*[?:\.\d*]+',text)
            for item in result:
                text = re.sub(item,',',text)
        # case 3: alphabet-digit
        if re.findall(r'[a-z|A-Z]+\-[\d]+',text):
            z = re.findall(r'[a-z|A-Z]+\-[\d]+',text)
            # return two things 
            for item in z:
                text = re.sub(item,',',text)
        # case 4: digit-alphabet
        if re.findall(r'[\d]+\-[a-z|A-Z]{3,}',text):
            z = re.findall(r'[\d]+\-[a-z|A-Z]{3,}',text)
            for item in z:
                text = re.sub(item,',',text)
        # case 5: prefix
        if re.findall(r'[a-z|A-Z]+(?:-[a-z|A-Z]+)+',text):
            #result = []
            z = re.findall(r'[a-z|A-Z]+(?:-[a-z|A-Z]+)+',text)
            for item in z:
                text = re.sub(item,',',text)
        # z = re.findall(r'^[\$][\d\.]$',word)[0]
        # parse all the digits 
        if re.findall(r'[\d]+',text):
            z = re.findall(r'[\d]+',text)
            for item in z:
                text = re.sub(item, ',', text)
            
        return text

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
    
    def _parse_special_tokens(self, raw):
        '''raw should already be split by whitespace
        '''
        for i, item in enumerate(raw):
            try:
                special_item = self.special_tokens(item)
                raw[i] = special_item
                self.special_toks.append(special_item)
            except UnboundLocalError:
                pass
        return raw
    
    def generate_ngrams(self, s, n):
        # cut special tokens
        s = self.cut_special_toks(s)
        # Convert to lowercases
        s = s.lower()
        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' zzz ', s)
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]
        for i,token in enumerate(tokens):
            if token in self.stop_words:
                tokens[i] = 'zzz'
                
        ngrams = zip(*[tokens[i:] for i in range(n)])
        #return ngrams
        rough_grams = [" ".join(ngram) for ngram in ngrams]
        result = []
        for item in rough_grams:
            if 'zzz' not in item:
                result.append(item)
        return result

    def parse(self,text):
        '''
        core function of the parser; we want to parse text to individual terms 
        this function will return a list of terms including special tokens and stopwords
        without punctuations
        '''
        for punc in self.punctuations:
            text = text.replace(punc,' ')
        # still have , . : _ - cuz they might appear in special tokens
        roughly_parsed = text.split(' ')
        other_puncs = [',','.','/','-','_','$','@',':']
       # now we have a set of terms that include stopwords, and special_terms (self.special_toks)
        special_parsed = self._parse_special_tokens(roughly_parsed)
        for i,term in enumerate(special_parsed):
            if term not in self.special_toks:
                for punc in other_puncs:
                    term = term.replace(punc,'')
                special_parsed[i] = term
        # remove empty, or tokens  
        processed = list(filter(lambda a:a!='',special_parsed))
        processed = list(filter(lambda a:a not in self.punctuations,processed))
        processed = list(filter(lambda a: a not in other_puncs,processed)) 
        # note that, the returning result include stopwords and also special tokens
        return processed
    
    def generate_phrases(self,text):
        
        self.two_grams = self.generate_ngrams(text,2)
        self.three_grams = self.generate_ngrams(text,3)

   
class Indexer:
    
    def __init__(self,stop_word_path):
        self.stop_words = self._set_of_stopwords(stop_word_path)
        self.single_term=[]
        self.stemmed = []
        self.term_positions = []
        self.phrases = []
        
        # set system argument for memory constraint
        #self.limit = 1000
        
    def process_dates(self,text):
        '''given some text, we want to process January 1, 2019 such as this format 
        '''
        ### case 1
        try:
            text = text.lower()
            months = {'january':'01','february':'02','march':'03','april':'04',\
                        'may':'05','june':'06','july':'80','august':'08','september':'09',\
                        'october':'10','november':'11','december':'12','jan':'01','feb':'02','mar':'03',\
                        'apr':'04',\
                        'may':'05','jun':'06','jul':'07','aug':'08','sep':'09',\
                        'oct':'10','nov':'11','dec':'12'}
            month_name_type = re.findall(r'[jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?]+\s+\d{1,2},\s*\d{4}',text)
        
            for item in month_name_type:
                result = re.sub(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2}),\s*(\d{4})', r'\1/\2/\3',item.rstrip())
                #plain = result.lower()
                k = result.split('/')
                # check if it is valid date
                mm = int(months[k[0]])
                dd = int(k[1])
                yy = int(k[2])
                correctDate = None
                try:
                    newdate = datetime(yy,mm,dd)
                    correctDate = True
                except ValueError:
                    correctDate = False
                    
                if correctDate:
        
                    new_res = result.replace(k[0],months[k[0]])
                    text = text.replace(item, new_res) 
                else:
                    # not valid date
                    continue
            # case 2
            num_type = re.findall(r'\d+\-\d+\-\d+',text)
            for item in num_type:
                #z = re.findall(r'(\d+)\-(\d+)\-(\d+)',word)[0]
                result = re.sub(r'(\d+)\-(\d+)\-(\d+)',r'\1/\2/\3',item.rstrip())
                # check if its valid date
                k = result.split('/')
                mm = int(k[0])
                dd = int(k[1])
                yy = int(k[2])
                correctDate = None
                try:
                    newdate = datetime.datetime(yy,mm,dd)
                    correctDate = True
                except ValueError:
                    correctDate = False
                if correctDate:
                    text = text.replace(item, result)
                else: 
                    continue 
        except:
            pass
            
        return text

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
    
    def strip_text(self,doc):
        '''given a document, stores its DOCNO, and stripped text into a dictionary
        '''
        d = {}
        DOCNO = re.findall(r"<DOCNO>([\s\S]*?)<\/DOCNO>",doc)[0]
        text = re.findall(r"<TEXT>([\s\S]*?)<\/TEXT>",doc)[0]
        # remove comments
        cleaned_text = re.sub(r"<!--[\s\S]*?-->", "", text)
        # remove \n
        new_txt = re.sub("\n", " ", cleaned_text)
        text = self.process_dates(new_txt)
        d['DOCNO'] = DOCNO
        # lower_case
        d['text'] = text.lower()
        return d
    
    def single_term_index(self,doc):
        '''accept a doc, by assumption, it will not exceed the memory constraint
        '''
        #single_term = {}
        # will include four parts: term, docID, tf
        stripped_dict = self.strip_text(doc)
        text = stripped_dict['text']
        # before parse it, we change all date formats
        tokenizer = Parser('stops.txt')
        terms = tokenizer.parse(text)
        single = {}
        # single term index without stopwords
        for term in terms:
            if type(term) == str and term not in self.stop_words:
                if term not in single:
                    single[term] = {'DOCNO':stripped_dict['DOCNO'],'tf':1}
                else:
                    single[term]['tf']+=1
            elif type(term) ==list:
                for k in term:
                    if k not in self.stop_words:
                        if k not in single:
                            single[k] = {'DOCNO':stripped_dict['DOCNO'],'tf':1}
                        else:
                            single[k]['tf']+=1
            else:
                pass
                #single[str(term)] = {'DOCNO':stripped_dict['DOCNO'],'tf':1}
        
        self.single_term+=sorted(single.items())

    
    def stem_index(self,doc):
        ''' accept a doc, return the stem index
            notice that we skip stop words and also we skip the special tokens 
        '''
        
        ps = PorterStemmer()
        #wnl = WordNetLemmatizer()
        stripped_dict = self.strip_text(doc)
        text = stripped_dict['text']
        tokenizer = Parser('stops.txt')
        terms = tokenizer.parse(text)
        stemmed_terms = {}
        for term in terms:
            try:
                if term not in self.stop_words:
                    stemmed_key = ps.stem(term)
                    if stemmed_key not in stemmed_terms:
                        stemmed_terms[stemmed_key] = {'DOCNO':stripped_dict['DOCNO'],'tf':1}
                    else:
                        stemmed_terms[stemmed_key]['tf']+=1
            except:
                # special tokens
                pass
        self.stemmed+=sorted(stemmed_terms.items())
    
    def positional_index(self,doc):
        ''' given some specific document, we want to get all the positional index 
            and store it as a list 
            here we do not skip stop_words or special tokens
        '''
        stripped_dict = self.strip_text(doc)
        text = stripped_dict['text']
        tokenizer = Parser('stops.txt')
        terms = tokenizer.parse(text)
        special_toks = tokenizer.special_toks
        positions = {}
        for i,term in enumerate(terms):
            if type(term) == str:
                if term not in positions:
                    positions[term] = {'DOCNO':stripped_dict['DOCNO'],'pos':[i]}
                else:
                    positions[term]['pos'].append(i)
            elif type(term) == list:
                # special tokens
                for k in term:
                    if k not in positions:
                        positions[k] = {'DOCNO':stripped_dict['DOCNO'],'pos':[i]}
                    else:
                        positions[k]['pos'].append(i)
            else:
                pass
            
        self.term_positions += sorted(positions.items())
        
    def phrase_index(self,doc):
        '''given some document, we want to get all the frequency index of all 2gram phrases and 3gram phrases
        '''
        stripped_dict = self.strip_text(doc)
        text = stripped_dict ['text']
        tokenizer = Parser('stops.txt')
        tokenizer.generate_phrases(text)
        two_grams = tokenizer.two_grams
        three_grams = tokenizer.three_grams
        all_phrases = two_grams + three_grams
        final_phrases = []
        grams = {}
        for item in all_phrases:
            if len(re.findall(r'[\d]+',item)) == 0:
                final_phrases.append(item)
        for phrase in final_phrases:
            try:
                if phrase not in grams:
                    grams[phrase] = {'DOCNO':stripped_dict['DOCNO'],'tf':1}
                else:
                    grams[phrase]['tf']+=1
            except:
                # special tokens
                pass
        self.phrases += sorted(grams.items())
    
    def parse_single_terms(self,files_path,files,limit):
        ''' 
        this function will parse everything in a single given collection
        '''
        path = './limit_'+str(limit)+'_single_term_index'
        # check if directory exists
        if not os.path.isdir(path):
            os.mkdir(path)
        for file in files:    
            coll = open(files_path+file)
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
                    self.single_term_index(mydoc)
                    #mydoc = ""
                    i+=1
                    
                    if len(self.single_term) >= limit:
                        # write to temporary pickle file 
                        #tr = dict(self.single_term)
                        to_write = sorted(self.single_term,key=lambda tup:tup[0])
                        _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_'+str(i)+'.txt')
                       # _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_'+str(i)+'.txt')
                        with open(_tempfile,'w') as f:    
                            for t in to_write:
                                f.write('#'.join(str(s) for s in t) + '\n')
                        f.close()
                        self.single_term = []
        
        if self.single_term!=[]:
            #tr = dict(self.single_term)
            to_write = sorted(self.single_term,key=lambda tup:tup[0])
            _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_end'+'.txt')
            with open(_tempfile,'w') as f:
                for t in to_write:
                    f.write('#'.join(str(s) for s in t) + '\n')
            f.close()
            self.single_term = []
    
    def parse_stem(self,files_path,files,limit):
        ''' 
        this function will parse everything in a single given collection
        '''
        path = './limit_'+str(limit)+'_stem_index'
        # check if directory exists
        if not os.path.isdir(path):
            os.mkdir(path)
        for file in files:    
            coll = open(files_path+file)
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
                    self.stem_index(mydoc)
                    #mydoc = ""
                    i+=1
                    
                    if len(self.stemmed) >= limit:
                        # write to temporary pickle file 
                        #tr = dict(self.stemmed)
                        to_write = sorted(self.stemmed,key=lambda tup:tup[0])
                        _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_'+str(i)+'.txt')
                        with open(_tempfile,'w') as f:    
                            for t in to_write:
                                f.write('#'.join(str(s) for s in t) + '\n')
                        f.close()
                        self.stemmed = []
            
        if self.stemmed!=[]:
            #tr = dict(self.stemmed)
            to_write = sorted(self.stemmed,key=lambda tup:tup[0])
            _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_end'+'.txt')
            with open(_tempfile,'w') as f:
                for t in to_write:
                    f.write('#'.join(str(s) for s in t) + '\n')
            f.close()
                       
            self.stemmed = []
            
    def parse_phrases(self,files_path,files,limit):
        ''' 
        this function will parse everything in a single given collection
        '''
        path = './limit_'+str(limit)+'_phrases'
        # check if directory exists
        if not os.path.isdir(path):
            os.mkdir(path)
        for file in files:    
            coll = open(files_path+file)
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
                    self.phrase_index(mydoc)
                    #mydoc = ""
                    i+=1
                    
                    if len(self.phrases) >= limit:
                        # write to temporary pickle file 
                        #tr = dict(self.phrases)
                        to_write = sorted(self.phrases,key=lambda tup:tup[0])
                        _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_'+str(i)+'.txt')
                        with open(_tempfile,'w') as f:    
                            for t in to_write:
                                f.write('#'.join(str(s) for s in t) + '\n')
                        f.close()
                        self.phrases = []
        
        if self.phrases!=[]:
            #tr = dict(self.phrases)
            to_write = sorted(self.phrases,key=lambda tup:tup[0])
            _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_end'+'.txt')
            with open(_tempfile,'w') as f:
                for t in to_write:
                    f.write('#'.join(str(s) for s in t) + '\n')
            f.close()           
            self.phrases = []
   
    def parse_positional(self,files_path,files,limit):
        ''' 
        this function will parse everything in a single given collection
        '''
        path = './limit_'+str(limit)+'_positional_index'
        # check if directory exists
        if not os.path.isdir(path):
            os.mkdir(path)
        for file in files:    
            coll = open(files_path+file)
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
                    self.positional_index(mydoc)
                    #mydoc = ""
                    i+=1
                    
                    if len(self.term_positions) >= limit:
                        # write to temporary pickle file 
                       # tr = dict(self.term_positions)
                        to_write = sorted(self.term_positions,key=lambda tup:tup[0])
                        _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_'+str(i)+'.txt')
                        with open(_tempfile,'w') as f:    
                            for t in to_write:
                                f.write('#'.join(str(s) for s in t) + '\n')
                        f.close()
                        self.term_positions = []
        
        if self.term_positions!=[]:
            #tr = dict(self.term_positions)
            to_write = sorted(self.term_positions,key=lambda tup:tup[0])
            _tempfile = str(path+'/'+file[:8]+'_'+file[9]+'_end'+'.txt')
            with open(_tempfile,'w') as f:
                for t in to_write:
                    f.write('#'.join(str(s) for s in t) + '\n')
            f.close()                      
            self.term_positions = []
            
      

################### Merging and Building IDF ##########################

def two_way_merge(file1,file2,output):
    ''' set a limit file, we want to merge all temp files 
    '''
    
    f1 = open(file1)
    f2 = open(file2)
    f1_line = f1.readline()
    f2_line = f2.readline()
    output_file = open(output,'w')
    while f1_line!= '':
        tp1 = f1_line.split('#')
        tp2 = f2_line.split('#')
        if tp1[0] < tp2[0]:
            output_file.write('#'.join(str(k) for k in tp1))
            f1_line = f1.readline()
        elif tp2[0] < tp1[0]:
            output_file.write('#'.join(str(k) for k in tp2))
            f2_line = f2.readline()
            if f2_line == '':
                break
        else:
            output_file.write('#'.join(str(k) for k in tp1))
            output_file.write('#'.join(str(k) for k in tp2))
            f1_line = f1.readline()
            f2_line = f2.readline()
            if f2_line == '':
                break
    while f2_line!='':
        output_file.write(f2_line)
        f2_line = f2.readline()
    f1.close()
    f2.close()
    output_file.close()

def merge_temps(limit,dirname):
    path = './limit_'+str(limit)+'_'+dirname+'/'
    if not os.path.isdir(path):
        os.mkdir(path)
    #path = './limit_1000_single_term_index/'
    cnt = 0
    all_files = os.listdir(path)

    while len(all_files)!=1:
        all_files = os.listdir(path)
        cnt +=1
        for i in range(int(len(all_files)/2)):
            file1 = path+all_files[i]
            file2 = path+all_files[-i-1]
            output = path+str(cnt)+'_merged_'+str(i)+'.txt'
            two_way_merge(file1,file2,output)
            os.remove(file1)
            os.remove(file2)    

def get_doc_length(collections,index_):
    ''' will return a dict of all documents and their terms
    we didnt exclude any stopwords or do any stemming here
    also, we didnt unlist specical_toks
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
                #doc_dict[d['DOCNO']] = {}
                if index_ == 'single_term_index':
                    indexer.single_term_index(mydoc)
                    doc_dict[d['DOCNO']] = len(indexer.single_term)
                    
                if index_ == 'stem_index':
                    indexer.stem_index(mydoc)
                    doc_dict[d['DOCNO']] = len(indexer.stemmed)
                if index_ == 'positional_index':
                    indexer.positional_index(mydoc)
                    doc_dict[d['DOCNO']] = len(indexer.term_positions)
                if index_ == 'phrases':
                    indexer.phrase_index(mydoc)
                    doc_dict[d['DOCNO']] = len(indexer.phrases)
                # for tpl in term_list:
                  #  lexi,info = tpl
                   # if index_ == 'positional_index':
                    #    doc_dict[d['DOCNO']][lexi] = info['pos']
                    #else:
                     #   doc_dict[d['DOCNO']][lexi] = info['tf']
                
            
    return doc_dict

def main(files_path,index_name,output_dir):
    #limit = sys.argv[1]
    # using default limit as 100000
    files = os.listdir(files_path)
    limit = 100000
    stage = {'single':'single_term_index','phrase':'phrases','stem':'stem_index','positional':'positional_index'}
    #with open('results.txt','a') as f:
    start = time.time()
    print('Starting indexing...'+'(default memory constraint: '+str(limit)+')'+'\n')
    
    indexer = Indexer('stops.txt')
    if index_name == 'positional':
        indexer.parse_positional(files_path,files,limit)
    if index_name == 'phrase':
        indexer.parse_phrases(files_path,files,limit)
    if index_name == 'stem':
        indexer.parse_stem(files_path,files,limit)
    if index_name == 'single':
        indexer.parse_single_terms(files_path,files,limit)
    
    print('Finished indexing.'+'\n')
    
    print('Starting merging...'+'\n')

    merge_temps(limit,stage[index_name])
    
    print('Finished merging. '+'\n')
    
    print('Starting building inverted index...'+'\n')

    path = './limit_'+str(limit)+'_'+str(stage[index_name])+'/'
    file = os.listdir(path)
    f = open(path+file[0])
    line = f.readline()
    lexicon = {}
    while line!='':
        term,pl = line.split("#")
        if term not in lexicon:
            lexicon[term] = [pl]
            line = f.readline()
        else:
            lexicon[term].append(pl)
            line = f.readline()
    
    run_time = time.time()-start
    print('Finished building inverted index.'+'\n'+'Running time: '+str(run_time)+'\n')
    # save the built results to output dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output = output_dir+'idf_'+str(limit)+'_'+str(stage[index_name])+'.pkl'
    pickle.dump(lexicon, open(output, "wb"))
    
    print('Backing-up document length files...'+'\n')
    # back up necessary document length files
    if not os.path.isdir('./document_length_dict/'):
        os.mkdir('./document_length_dict/')
    doc_len = get_doc_length(files,stage[index_name])
    output = './document_length_dict/'+index_name+'_doc_length.pkl'
    pickle.dump(doc_len, open(output, "wb"))
    print('Finished backing up document length files.'+'\n')
    # remove folder for holding temporary files

    
if __name__ == "__main__":
    files_path = sys.argv[1]
    index_name = sys.argv[2]
    output_dir = sys.argv[3]
    main(files_path,index_name,output_dir)






