# Implement Transformer In Keras 
*attention is all your need*

Transformer is an important NLP area break-through from paper *[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)* in 2017. It uses attention mechansim to replace recurrent and convolutionary layers in sequence deep learning.
Encoder-Decoder architect is the classic framework for seq2seq learning, such as neural machine translate, which maps a sentence to another sentence.
Before transformer came out, the standard way of encoder-decoder is using 

* Embedding layers to represent discrete words into low dimensional dense vectors
* LSTM/GRU as encoder & decoder, to learn the sequential patterns
* decoder-encoder attention mechansim to give different priority on encoder outputs 

Transformer, instead has:

* self-attention: to extract features from encoder/decoder itself.
* multi-head attention: increase the parallism of the model
* position-encoding: to capture the position of tokens, since no convo or recurrent layers are applied
* residual-connection: to increase model capability of generality, allow deeper model.

## scaled dot product attention

![](https://raw.githubusercontent.com/6chaoran/data-story/master/nlp/transformer/attention_head.JPG)

```python
def scaled_dot_product_attention(Q,K,V,k,mask = False):
    """
    Q: query => split decoder input (batch_size, num_of_heads, length, latent_dim_for_head)
    K: key => encoder input
    V: value => same as K for machine translation
    k: model hidden dimension
    
    Return:
        attention output (batch, h, length, latent_dim_for_head)
    """
    _, h, length, _ = Q.shape.as_list()
    
    matmul_1 = Dot(axes=-1)
    matmul_2 = Dot(axes=-1)
    softmax = Activation(activation='softmax')
    scale = Lambda(lambda x: x / tf.sqrt(tf.constant(k,dtype=tf.float32)))
    swop_axis = Lambda(lambda x: permute_dimensions(x, (0,1,3,2)))
    
    logit = matmul_1([Q,K])
    logit = scale(logit)
    if mask:
        add_mask = Lambda(lambda x: x + get_mask(length))
        logit = add_mask(logit)
    logit = softmax(logit)
    logit = matmul_2([logit,swop_axis(V)])
    return logit

```

## multi-head attention

```python
def multi_head_attention(Q,K,V,k,h,mask = False):
    """
    h: number of attention heads
    """
    linear_q = Dense(k)
    linear_k = Dense(k)
    linear_v = Dense(k)
    linear_output = Dense(k)
    
    # split heads function split the latent_vector into h head
    # input tensor (batch, length, latent_dim) => (batch, heads, length, latent_dim//heads)
    Q = split_heads(linear_q(Q),h)
    K = split_heads(linear_k(K),h)
    V = split_heads(linear_v(V),h)
    
    heads = scaled_dot_product_attention(Q,K,V,k,mask=mask)
    heads = combine_heads(heads)
    heads = linear_output(heads)
    return heads
```

## position-wise feed forward

![](https://raw.githubusercontent.com/6chaoran/data-story/master/nlp/transformer/feed-forward.JPG)

```python
def position_wise_feed_foward(ff,filter_size,k):
    
    ff = TimeDistributed(Dense(filter_size, activation='relu'))(ff)
    ff = TimeDistributed(Dense(k))(ff)
    
    return ff
```

## skip connection

```python
def add_and_norm(input1, input2):
    out = Add()([input1,input2])
    out = BatchNormalization()(out)
    return out
```

## position encoding

![](https://raw.githubusercontent.com/6chaoran/data-story/master/nlp/transformer/positional-encoding.JPG)

```python
def positional_encoding(length, k):
    num_timescales = k // 2
    max_timescale = 10000
    min_timescale = 1
    pos = K.arange(length,dtype='float32')
    log_timescale_increment = (
          K.log(float(max_timescale) / float(min_timescale)) /
          (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * K.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(pos, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal
```

## encoder block

![](https://raw.githubusercontent.com/6chaoran/data-story/master/nlp/transformer/transformer-architecture.JPG)

```python
def encoder_block(Q,k,h, filter_size):
    
    # attention sub-layer
    attention = multi_head_attention(Q,Q,Q,k,h)
    attention = add_and_norm(attention, Q)
    # feed forward sub-layer
    ff = position_wise_feed_foward(attention, filter_size, k)
    ff = add_and_norm(ff, attention)
    
    return ff
```
## decoder block

```python
def decoder_block(Q,k,h,N,filter_size,encoder_output):
    
    # self attention sub-layer
    self_attention = multi_head_attention(Q,Q,Q,k,h, mask=True)
    self_attention = add_and_norm(self_attention, Q)
    
    # encoder decoder attention sub-layer
    enc_dec_attention = multi_head_attention(self_attention, encoder_output, encoder_output, k, h)
    enc_dec_attention = add_and_norm(enc_dec_attention, self_attention)
    
    # feed forward sub-layer
    ff = position_wise_feed_foward(enc_dec_attention,filter_size,k)
    ff = add_and_norm(ff, enc_dec_attention)
    
    return ff
```