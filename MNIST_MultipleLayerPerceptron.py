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


# 파라미터
# 학습률
# Epoch - 학습 횟수
# batch 크기 - 한번에 학습하는 데이터 갯수
learning_rate = 0.001
training_epochs = 20
batch_size = 100


# 입력층, 은닉층, 출력층의 파라미터
# 첫번째 은닉층
# 두번째 은닉층
# 입력층 - 28 * 28
# 출력층 - 0 ~ 9
n_hidden_1 = 300
n_hidden_2 = 300
n_input = 784
n_classes = 10


# Placeholder > session의 값들을 변수로 사용 가능
# KNN 을 하기 위해서 x 값들의 거리를 확인한다
# x : training data의 784개의 픽셀(28 * 28)
# y : training data의 10개의 숫자 (0 ~ 9)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Multilayer Perceptron 모델
# Relu를 사용(sigmoid의 error vanishing을 해결하는 활성함수)
def multilayer_perceptron(x, weights, biases):
    # 첫번째 은닉
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 두번째 은닉
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 출력층
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# 가중치 초기화 및 저장
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Tensorflow를 이용한 작업
# 다중 퍼셉트론 - 딥러닝
# pred : 위에서 정의한 모델 - 다중 퍼셉트론
# correct_prediction : 0 ~ 9 까지 숫자를 맞추었는지 여부
# accuracy : 정확도
pred = multilayer_perceptron(x, weights, biases)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# cross entropy : http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220554852626
# adam optimizer로 오차 최소화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Tensorflow 세션 실행
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # 훈련 사이클
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # 모든 batch를 학습
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # 평균 오차를 구한다
            avg_cost += c / total_batch

        # 모든 batch 학습 후 평균 오차 출력
        print("Epoch:", '%2d' % (epoch+1), "| cost=", \
            "{:.3f}".format(avg_cost))

    # 정확도 출력
    print("\n정확도:", "{:.2f}".format(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})*100),"%")
