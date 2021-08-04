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

def forward_segment(query:str, dic:set):
    """
    使用前向匹配识别分词，从前往后匹配字典中的词典，依次选着最长的词语
    query: str
    dic: ("entity")
    """
    out_map, i, q_len = [], 0, len(query)
    while i < q_len:
        max_word = query[i]
        for j in range(i + 1, q_len + 1):
            cur_w = query[i:j]
            if cur_w in dic:
                max_word = cur_w
        out_map.append(max_word)
        i += len(max_word)
    return out_map

def backward_segment(query:str, dic:set):
    """
    使用后向匹配识别句子中的实体，非分词，从后往前匹配字典中的词典，依次选着最长的词语
    query: str
    dic: {"entity": "entity_type"}
    """
    out_map, i = [], len(query) - 1
    while i >= 0:
        max_word = query[i]
        for j in range(i - 1, -1, -1):
            cur_w = query[j:i + 1]
            if cur_w in dic:
                max_word = cur_w
        out_map.append(max_word)
        i -= len(max_word)
    return out_map

def bi_segment(query:str, dic:set):
    """
    双向最大匹配分词：
    1. 对于一条语句匹配出来的词也就越少，相对准确性也就会越高。
    2. 如果词数目一致，那么有两种：
        a.分词结果相同，就说明没有歧义，可返回任意一个。
        b.分词结果不同，返回其中单字较少的那个。
    
    """
    forward = forward_segment(query, dic)
    backward = backward_segment(query, dic)
    #返回分词数较少者
    if len(forward) > len(backward):
        return backward
    elif len(forward) < len(backward):
        return forward
    else:
        #大小一致，返回任意一个
        return forward
    pass

def forward_match(query:str, dic:dict):
    """
    使用前向匹配识别句子中的实体，非分词，从前往后匹配字典中的词典，依次选着最长的词语
    query: str
    dic: {"entity": "entity_type"}
    """
    out_map, i, q_len = [], 0, len(query)
    while i < q_len:
        max_word = query[i]
        for j in range(i + 1, q_len + 1):
            cur_w = query[i:j]
            if cur_w in dic:
                max_word = cur_w
        if max_word in dic:
            out_map.append([max_word, dic[max_word]])
        i += len(max_word)
    return out_map

def backward_match(query:str, dic:dict):
    """
    使用后向匹配识别句子中的实体，非分词，从后往前匹配字典中的词典，依次选着最长的词语
    query: str
    dic: {"entity": "entity_type"}
    """
    out_map, i = [], len(query) - 1
    while i >= 0:
        max_word = query[i]
        for j in range(i - 1, -1, -1):
            cur_w = query[j:i + 1]
            if cur_w in dic:
                max_word = cur_w
        if max_word in dic:
            out_map.append([max_word, dic[max_word]])
        i -= len(max_word)
    return out_map

def bi_match(query, dic):
    """
    双向最大匹配原则：
    1. 对于一条语句匹配出来的词也就越少，相对准确性也就会越高。
    2. 如果词数目一致，那么有两种：
        a.分词结果相同，就说明没有歧义，可返回任意一个。
        b.分词结果不同，返回其中单字较少的那个。
    
    """
    forward = forward_match(query, dic)
    backward = backward_match(query, dic)
    #返回分词数较少者
    if len(forward) > len(backward):
        return backward
    elif len(forward) < len(backward):
        return forward
    else:
        #大小一致，返回任意一个
        return forward
    pass

if __name__ == '__main__':
    # 南京市长江大桥
    word_dic = {'南京': 'city', '南京市': 'city', '长江大桥': 'location', '长江': 'location', '市长': 'post', '南京市长': 'post'}
    print("forward_match:", forward_match('南京市长江大桥', word_dic))
    print("backward_match:", backward_match('南京市长江大桥', word_dic))
    print("forward_segment:", forward_segment('南京市长江大桥', word_dic.keys()))
    print("backward_segment:", backward_segment('南京市长江大桥', word_dic.keys()))
    print("bi_segment:", bi_segment('南京市长江大桥', word_dic.keys()))
    
