#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2021/08/03 23:53:52
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
from collections import defaultdict

class leftMax(object):
    def __init__(self,dict_path):
        self.dictionary = set() #定义字典
        self.maximum = 0 #最大匹配长度
        
        with open(dict_path,'r',encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line.split('\t')[1])
                if len(line) > self.maximum:
                    self.maximum = len(line)
                    
    def cut(self,text):
        result = []
        length = len(text)
        index = 0
        while length > 0:
            word = None
            for size in range(self.maximum,0,-1):
                if length - size < 0:
                    continue
                piece = text[index:index+size]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    length -= size
                    index += size
                    break
            if word is None:
                length -= 1
                result.append(text[index])
                index += 1
        return result

def forward_segment(query:str, dic:dict):
    """
    使用前向匹配识别句子中的实体，非分词
    query: str
    dic: {"entity": "entity_type"}
    """
    
    pass

if __name__ == '__main__':
    word_dic = {'北京大学': 'org', '算法工程师': 'post'}
    
