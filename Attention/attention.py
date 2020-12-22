#!/usr/bin/env python
# encoding=utf-8
'''
@Time    :   2019/10/06 23:47:08
@Author  :   Zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   attention有多种实现方式，同时也有多种用法。NLP中：
1. 对于分类问题，句子中每个token对于类别标签贡献是不一样的，所以存在句子中每个token对于类别的attention权重。
2. 对于NER问题，产生每个token的标签时，对于其上下文的词语也有一个attention权重，
'''

# here put the import lib
import collections
import copy
import json
import math
import re
import six
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
    attn_emb_1 = tf.reshape(tf.matmul(tf.reshape(
        input_emb, [-1, q_hidden_size]), attn_w), [-1, seq_len, attn_hidden_size])
    # batch_size * 1 * attn_hidden_size
    attn_emb_2 = tf.expand_dims(tf.matmul(kv_emb, attn_u), 1)
    # batch_size * seq_len * attn_hidden_size
    attn_emb = tf.tanh(tf.add(attn_emb_1, attn_emb_2))
    # batch_size * seq_len * 1
    attn_logits = tf.reshape(tf.matmul(tf.reshape(
        attn_emb, [-1, attn_hidden_size]), attn_v), [-1, seq_len, 1])
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
    # attn_y = core_rnn_cell._linear(attn_input, attn_size, True)
    attn_y = tf.layers.dense(attn_input, attn_size)
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


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.
    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.
    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.
    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).
    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
