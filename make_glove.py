# 用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度
import gensim
import os
import shutil
import hashlib
from sys import platform
from gensim.models import KeyedVectors
import pickle

# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r',encoding='utf-8')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r',encoding='utf-8') as fin:
        with open(outfile, 'w',encoding='utf-8') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)

def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename
load('glove.6B.300d.txt')
wvmodel = KeyedVectors.load_word2vec_format('glove_model.txt',binary=False,encoding='utf-8')
filename = save_variable(wvmodel,'dict.txt')
wvmodel.similar_by_vector

