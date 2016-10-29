#_*_ coding: utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 생성
# 눈이 올 확률
# b = x0 = 1
# x1 = 비가 올 확률( 0 ~ 10 )
# x2 = 온도 수치 ( 0 ~ 10 )
# y = 눈이 오는지 안오는지

# 데이터 생성
# 100개의 데이터
num_datas = 100

data = []


# 100개의 데이터
# x1 > 5 && x2 < 5 이면 눈이 온다는 알고리즘으로 데이터를 생성
for i in range(num_datas):

    x1 = np.random.uniform(0.0,10.0)
    x2 = np.random.uniform(0.0,10.0)
    if x1 > 5.0 and x2 < 5.0:
        y = 1
        # 눈이 온다 - 파란색 점
        plt.plot(x1, x2, 'bo')
    else:
        y = 0
        # 눈이 안온다 - 빨간색 점
        plt.plot(x1, x2, 'ro')
    data.append([1.0,x1,x2,y])


# 한번 그려봅니다
plt.show()

# Tensorflow에서 쓰기 위한 데이터 가공
data_transpose = np.transpose(data)

x_data = np.float32(data_transpose[0:-1])
y_data = np.array(data_transpose[-1])[np.newaxis]

# Placeholder > session의 값들을 변수로 사용 가능
# 재활용성 상승 및 좀 더 유연한 프로그래밍 가능 - 앞으로 계속 사용 예정
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# Tensorflow를 이용한 작업
# 로지스틱 회귀
# pred = sigmoid(W*X) = 로지스틱 함수의 꼴 (sigmoid는 로지스틱 함수의 특별한 케이스)
# W : weight > 초기화를 -1.0과 1.0 사이의 값으로 랜덤하게 정한다
W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
pred = tf.nn.sigmoid(tf.matmul(W,X))

# cost or loss function
# 기존의 것을 사용하면 global minima가 아닌 local minima에 빠짐 - 로지스틱 함수가 선형식이 아니기 때문
# 다른 function을 사용 = - Y*log(pred) - (1 - Y)*log(1-pred)
# 자세한 설명 링크 : http://jsideas.net/octave/2016/01/04/ml_logistic_regression.html
# optimizer = GD Optimizer
# learning rate = 0.1
# 너무 작으면 느리게 학습하고
# 너무 크면 발산해 버림
# auc = ROC 커브에 의한 AUC 값
loss = -tf.reduce_mean(Y*tf.log(pred) + (1-Y)*tf.log(1-pred))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
auc= tf.contrib.metrics.streaming_auc(tf.nn.sigmoid(tf.matmul(W, X)), Y,curve='ROC')

# Tensorflow 세션 실행
init = tf.initialize_all_variables()
local_init = tf.initialize_local_variables()

sess = tf.Session()
sess.run(init)
sess.run(local_init)

# 1000번의 학습을 반복
# 각 100 번의 학습마다 결과를 출력해 본다
auc_data = []
step_data = []
for step in range(1000):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 100 == 0:
        percentages = sess.run(pred, feed_dict={X:x_data, Y:y_data})
        print("예측: " + str(percentages))
        train_auc,_ = sess.run(auc, feed_dict={X:x_data, Y:y_data})
        print("예측 정확도 : " + str(train_auc))
        auc_data.append(train_auc)
        step_data.append(step/100)
plt.plot(step_data,auc_data)
plt.show()