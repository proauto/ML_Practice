#_*_ coding: utf-8 _*_
# Tesorflow 에서 제공하는 Mnist 데이터를 쉽게 가져오는 tutorial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# Mnist 데이터 셋

# mnist.train(55,000 개의 훈련 데이터)
# mnist.validation(5000개의 검증 데이터)

# 데이터.images 혹은 데이터.labels 와 같이 사용
# 각각의 이미지는 28*28(784) pixel 이고 label은 0~9 까지의 숫자
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("\nMNIST 데이터 전송 완료\n")

# Placeholder > session의 값들을 변수로 사용 가능
# KNN 을 하기 위해서 x 값들의 거리를 확인한다
# x_i : training data 전부의 784개의 픽셀
# x_e : test data 한개 784개의 픽셀
x_i = tf.placeholder(tf.float32, [None, 784])
x_e = tf.placeholder(tf.float32, [784])


# Tensorflow를 이용한 작업
# distance : K-Nearest Neighbor - 오차 거리를 구한다(유클리드 거리)
# pred : 오차가 최소인 index

distance = tf.reduce_sum(tf.abs(tf.add(x_i, tf.neg(x_e))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

# 학습을 위해서 Batch size를 나누는 것이 아니라 기준 데이터들의 개수를 Batch size로 잡음
# 100개에 대해 테스트 해본다
# 정확도를 초기화
batch_xs, batch_ys = mnist.train.next_batch(5000)
test_xs, test_yx = mnist.test.next_batch(1000)
accuracy = 0.

# Tensorflow 세션 실행
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#   테스트 케이스마다 한번씩 비교를 해주어야 한다.
for i in range(len(test_xs)):
    # training set에서 가장 가장 근접한 지점의 index(batch size개의 기준과 테스트 케이스를 비교)
    nn_index = sess.run(pred, feed_dict={x_i: batch_xs, x_e: test_xs[i, :]})

    #nn_index의 라벨값과 실제 라벨값을 비교한다.
    print("테스트 횟수 : ", i)
    print("실제값 : ", np.argmax(test_yx[i]))
    print("예측값 : ", np.argmax(batch_ys[nn_index]))

    # 예측도 파악
    # KNN은 비교 데이터에서 가장 가까운 것을 찾는 것이므로 매번 확률을 갱신해야한다(가중치를 찾는게 아니다)
    # 가장 가까운 것이 무엇이 될지 모름
    if np.argmax(batch_ys[nn_index]) == np.argmax(test_yx[i]):
        accuracy += 1./len(test_xs)

print("예측 정확도 : ", round(accuracy*100, 2)," %")