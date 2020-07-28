""" download data
!wget http://www.manythings.org/anki/cmn-eng.zip
!unzip data/cmn-eng.zip
!mv cmn-eng.txt data/
"""

import numpy as np
import pandas as pd
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
import jieba

def clean_data(): 
	df = pd.read_csv('data/cmn.txt',sep='\t',header=None, names=['eng','chn'])
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs
 
def clean_eng(line):
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.split()
    # convert to lowercase
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(table) for word in line]
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]    
    
    return line

def clean_chn(line):
    puncts = ['。','，','！','？']
    # tokenize on white space
    line = jieba.cut(line, cut_all=False)
    # remove punctuation from each token
    line = [word for word in line if word not in puncts]
    # remove non-printable chars form each token
    #line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    #line = [word for word in line if word.isalpha()]    
    
    return line

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = [' '.join(clean_eng(pair[0])),' '.join(clean_chn(pair[1]))]
		cleaned.append(clean_pair)
	return array(cleaned)
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)


if __name__ == '__main__':
	
	# load dataset
	filename = './data/cmn.txt'
	doc = load_doc(filename)
	# split into english-german pairs
	pairs = to_pairs(doc)
	# clean sentences
	clean_pairs = clean_pairs(pairs)
	# save clean pairs to file
	save_clean_data(clean_pairs, 'english-chinese.pkl')
	# spot check
	for i in range(10):
		print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))