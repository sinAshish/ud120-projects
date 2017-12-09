#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:05:13 2017

@author: ashish
"""
from nltk.corpus import stopwords
sw=stopwords.words('english')
print len(sw)

from nltk.stem.snowball import SnowballStemmer
st=SnowballStemmer('english')
print st