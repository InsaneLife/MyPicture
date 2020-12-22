#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2019/10/06 23:47:08
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   attention有多种实现方式，同时也有多种用法。NLP中：
1. 对于分类问题，句子中每个token对于类别标签贡献是不一样的，所以存在句子中每个token对于类别的attention权重。
2. 对于NER问题，产生每个token的标签时，对于其上下文的词语也有一个attention权重，
'''

# here put the import lib
import tensorflow as tf

def perceptron_attention(input_emb, kv_emb, attn_hidden_size=None):
    """
    多层感知机的attention
    过滤出q中与kv相关的信息, 即 score(q, k) * q
    q = input_emb : batch_size * seq_len * hidden_size
    k = v = kv_emb: batch_size * hidden_size
    score(q, k) = V * tanh(W * q + U * k)
    """
    q_shape = input_emb.shape
    batch_size, seq_len, q_hidden_size = q_shape[0].value, q_shape[1].value, q_shape[2].value
    kv_hidden_size = kv_emb.shape[-1].value

    if not attn_hidden_size:
        attn_hidden_size = q_hidden_size

    attn_w = tf.get_variable(name="attn_w", shape=[q_hidden_size, attn_hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    attn_u = tf.get_variable(name="attn_u", shape=[kv_hidden_size, attn_hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    attn_v = tf.get_variable(name="attn_v", shape=[attn_hidden_size, 1],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

    # batch_size * seq_len * attn_hidden_size
    attn_emb_1 = tf.reshape(tf.matmul(tf.reshape(input_emb, [-1, q_hidden_size]), attn_w), [-1, seq_len, attn_hidden_size])
    # batch_size * 1 * attn_hidden_size
    attn_emb_2 = tf.expand_dims(tf.matmul(kv_emb, attn_u), 1)
    # batch_size * seq_len * attn_hidden_size
    attn_emb = tf.tanh(tf.add(attn_emb_1, attn_emb_2))
    # batch_size * seq_len * 1
    attn_logits = tf.reshape(tf.matmul(tf.reshape(attn_emb, [-1, attn_hidden_size]), attn_v), [-1, seq_len, 1])
    attn_weights = tf.nn.softmax(attn_emb, 1)

    word_outputs = tf.multiply(input_emb, attn_weights)
    sen_output = tf.reduce_sum(word_outputs, axis=1)

    return word_outputs, sen_output, attn_weights

def scaled_dot_prod_attention(input_emb, kv_emb):
    """
    缩放点积注意力
    score(q, k) * v, 使用v中的信息更新q中的信息
    q = input_emb:  batch_size * seq_len_q * hidden_size
    k = v = kv_emb: batch_size * seq_len_k * hidden_size
    """
    # batch_size * seq_len_q * seq_len_k
    dot_prod = tf.matmul(input_emb, kv_emb, transpose_b=True)
    dims = tf.cast(tf.shape(input_emb)[-1], tf.float32)
    scaled_attn_logits = dot_prod / tf.sqrt(dims)
    attn_weights = tf.nn.softmax(scaled_attn_logits, -1)

    # batch_size * seq_len_q * hidden_size
    word_outputs = tf.matmul(attn_weights, kv_emb)
    sen_output = tf.reduce_sum(word_outputs, axis=1)

    return word_outputs, sen_output, attn_weights


def slot_attention(embeddings, reduce_axis):
    """
    通过attention得到句子级别的embedding
    :param embeddings: [batch_size, seq_length, embedding_size]
    :param reduce_axis: [2] 生成attention后word级别的embedding; [1, 2] 生成attention后query级别的embedding
    :return:
    """
    attn_input = embeddings
    attn_output = embeddings

    attn_size = embeddings.get_shape()[2].value

    origin_shape = tf.shape(embeddings)

    attn_w = tf.get_variable("attn_w", [1, 1, attn_size, attn_size])
    attn_v = tf.get_variable("attn_v", [attn_size])
    # hidden_conv: [batch_size, seq_length, 1, embedding_size]
    hidden_conv = tf.expand_dims(attn_output, 2)
    hidden_features = tf.nn.conv2d(hidden_conv, attn_w, [1, 1, 1, 1], "SAME")
    hidden_features = tf.reshape(hidden_features, origin_shape)
    # hidden_features: [batch_size, 1, seq_length, embedding_size]
    hidden_features = tf.expand_dims(hidden_features, 1)

    attn_input = tf.reshape(attn_input, [-1, attn_size])
    attn_y = core_rnn_cell._linear(attn_input, attn_size, True)
    attn_y = tf.reshape(attn_y, origin_shape)
    # attn_y: [batch_size, seq_length, 1, embedding_size]
    attn_y = tf.expand_dims(attn_y, 2)
    # attn_s: [batch_size, seq_length, seq_length]
    attn_s = tf.reduce_sum(attn_v * tf.tanh(hidden_features + attn_y), [3])
    attn_a = tf.nn.softmax(attn_s)
    # attn_a: [batch_size, seq_length, seq_length, 1]
    attn_a = tf.expand_dims(attn_a, -1)
    # attn_a * hidden_conv: [batch_size, seq_length, seq_length, embedding_size]
    # attn_d: reduce_axis = [2]: [batch_size, seq_length, embedding_size]
    #         reduce_axis = [1, 2]: [batch_size, embedding_size]
    attn_d = tf.reduce_sum(attn_a * hidden_conv, reduce_axis)

    return attn_d