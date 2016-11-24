# reference : http://hunkim.github.io/ml/

import tensorflow as tf
import numpy as np


# 데이터 생성
# char_rdic : Hello World 출력을 위한 알파벳 카테고리 # id -> char
# char_dic : 카테고리를 알파벳에서 숫자로 바꾼다. # char -> id
# x_data : RNN의 입력으로 넣어줄 데이터 - HelloWorl
# sample : 정답인 HelloWorld를 indexing 해준다.
char_rdic = ['H','e','l','o','W','r','d']
char_dic = {w: i for i, w in enumerate(char_rdic)}
x_data = np.array([[1, 0, 0, 0, 0, 0, 0], #H
                   [0, 1, 0, 0, 0, 0, 0], #e
                   [0, 0, 1, 0, 0, 0, 0], #l
                   [0, 0, 1, 0, 0, 0, 0], #l
                   [0, 0, 0, 1, 0, 0, 0], #o
                   [0, 0, 0, 0, 1, 0, 0], #W
                   [0, 0, 0, 1, 0, 0, 0], #o
                   [0, 0, 0, 0, 0, 1, 0], #r
                   [0, 0, 1, 0, 0, 0, 0], #l
                    ],
                  dtype='f')
sample = [char_dic[c] for c in "HelloWorld"]  # to index


# 기본 설정
# rnn_size : RNN 입출력의 크기
# time_step_size : 출력을 반복하는 횟수 ('HelloWorl' -> predict 'elloWorld')
rnn_size = len(char_dic)
time_step_size = 9
learning_rate = 0.03


# RNN model
# rnn_cell : RNN 셀
# state : hidden state 초기화
# X_split : 입력의 크기
# ouputs, state : 출력과 갱신된 hidden state
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([1, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)


# seq2seq.py를 이용하기 위한 초기화
# logits : list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets : list of 1D batch-sized int32 Tensors of the same length as logits
# weights : list of 1D batch-sized float-Tensors of the same length as logits
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * 1])


# cost : Tensorflow에서는 sequence_loss_by_example를 기본으로 제공한다
# train_op : adam optimizer로 오차 최소화
cost = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example([logits],[targets],[weights]))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Tensorflow 세션 실행
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 100 번 학습
for i in range(100):
    sess.run(train_op)
    result = sess.run(tf.arg_max(logits, 1))

    output = ''.join([char_rdic[t] for t in result])
    print(i+1, "/ 100 ","H" + output)