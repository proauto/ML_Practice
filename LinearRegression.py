#_*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# 데이터 생성
# 100개의 데이터
num_datas = 100

data = []

# 100개의 y = x 그래프를 기준으로 0~1까지 오차를 가진 데이터
for i in range(num_datas):
    x = np.random.normal(0.0, 3.0)
    y = x*1 + np.random.normal(0.0, 1)
    data.append([x,y])
    print(i)

x_data = [v[0] for v in data]
y_data = [v[1] for v in data]

# 한번 그려봅니다
plt.plot(x_data,y_data,'bo')
plt.show()

# Tensorflow를 이용한 작업
# 선형 회귀
# y = W * x + b
# W : weight > 초기화를 -1.0과 1.0 사이의 값으로 랜덤하게 정한다
# b : bias > 초기값은 0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# cost or loss function = 오차의 RMS
# optimizer = GD Optimizer
# 여기서 0.01은 학습률
# 너무 작으면 느리게 학습하고
# 너무 크면 발산해 버림
loss = tf.reduce_mean(tf.square(y- y_data))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Tensorflow 세션 실행
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 10번의 학습을 반복
# 각 학습마다 결과를 출력해 본다
for step in range(10):
    sess.run(train)
    plt.plot(x_data, y_data, 'bo')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'r')
    plt.legend()
    plt.show()

