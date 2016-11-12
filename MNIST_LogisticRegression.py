#_*_ coding: utf-8 _*_
# Tesorflow 에서 제공하는 Mnist 데이터를 쉽게 가져오는 tutorial
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Mnist 데이터 셋

# mnist.train(55,000 개의 훈련 데이터)
# mnist.validation(5000개의 검증 데이터)

# 데이터.images 혹은 데이터.labels 와 같이 사용
# 각각의 이미지는 28*28(784) pixel 이고 label은 0~9 까지의 숫자
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("\nMNIST 데이터 전송 완료\n")

# Placeholder > session의 값들을 변수로 사용 가능
# x : 784개의 픽셀
# y : 10개의 라벨(0 ~ 9)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Tensorflow를 이용한 작업
# 로지스틱 회귀
# pred = softmax(W*X + b) = 여러개의 sigmoid 값들을 모두 더한 값으로 나누어 전체 label 확률의 합이 1이 되도록 하는 것(가장 큰 확률 Label이 선택됨)
# W,b : weight = 0 으로 초기화
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

# http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220554852626
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 훈련 step은 cross_entropy를 최소화하는 수준으로 설정
# 정답 예측은 카테고리(0~9) 답변이기에 같은지 다른지만 확인
# 예측 정확도는 정답 예측의 정도를 RMS 한 값
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Tensorflow 세션 실행
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  # Batch Size를 설정하는 것으로 한번에 한개씩 학습하는 것이 아니라 여러 데이터 셋을 한번에 학습할 수 있다
  # 장점 :
  # Batch Size를 설정하고 반복마다 다음 train set을 넘겨줌
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

print("예측 정확도 : " + str(round(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})*100,2)) + "%")