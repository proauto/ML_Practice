#_*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 데이터 생성
# 100개의 데이터
num_datas = 100

data = []

# 100개의 y = x1 + x2 그래프를 기준으로 0~1까지 오차를 가진 데이터
for i in range(num_datas):
    x1 = np.random.normal(0.0, 5.0)
    x2 = np.random.normal(0.0, 5.0)
    y = x1*1 + x2*1 + np.random.normal(0.0, 1.0)
    data.append([x1,x2,y])
    print(i)

x1_data = [v[0] for v in data]
x2_data = [v[1] for v in data]
y_data = [v[2] for v in data]

# 한번 그려봅니다
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1_data, x2_data, y_data, 'r', marker='o')

ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('Y Label')

plt.show()


# Tensorflow를 이용한 작업
# 선형 회귀
# y = W1 * x1 + W2 * x2 + b
# W1, W2 : weight > 초기화를 -1.0과 1.0 사이의 값으로 랜덤하게 정한다
# b : bias > 초기값은 0
W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W1 * x1_data + W2 * x2_data + b

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
# 변수가 x1, x2로 두 개이므로 평면으로 표현된다.
for step in range(10):
    sess.run(train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1_data, x2_data, y_data, 'r', marker='o')

    x1 = np.arange(-10, 10, 0.25)
    x2 = np.arange(-10, 10, 0.25)
    X1, X2 = np.meshgrid(x1, x2)
    Y = sess.run(W1) * X1 + sess.run(W2) * X2 + sess.run(b)

    surf = ax.plot_surface(X1, X2, Y, rstride=2, cstride=2, cmap=cm.RdPu, linewidth=1, antialiased=True)
    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('Z Label')

    plt.show()


