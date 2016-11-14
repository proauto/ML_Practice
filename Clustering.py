import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
# 세 부분으로 나누어져서 퍼져있는 데이터

# 데이터 생성
# 3000개의 데이터
# 각각 1/3 확률로 (-1,-1), (9,9), (7,-3) 을 기준으로
# 정규 분포로 데이터를 생성 > 퍼지는 정도는 약간의 차이를 둠

num_datas = 3000
data = []

for i in range(num_datas):
    choice = np.random.random()

    if choice < 1/3:
        x = np.random.normal(-1.0,1.0)
        y = np.random.normal(-1.0,1.5)
    elif choice < 2/3:
        x = np.random.normal(9.0, 1.0)
        y = np.random.normal(9.0, 2.0)
    else:
        x = np.random.normal(7.0, 2.0)
        y = np.random.normal(-3.0, 1.0)

    plt.plot(x, y, 'bo')
    data.append([x, y])


# 한번 그려봅니다
print("데이터 생성 완료")
plt.show()

# K- Clustering
K=3

# 생성한 데이터들을 Tensor로 만들어주고
# 랜덤으로 중심값(centroid)을 K개 고른다.
vectors = tf.constant(data)
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0, 0],[K, -1]))


# 서로 비교를 위해 차원 확장 > 2차원에서는 크기가 다르기 때문에 비교가 불가 > 3차원으로 확장
# (3000, 2), (K, 2) > (n, 3000, 2), (K, n, 2)
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroids, 1)



# cost or loss function = 오차의 RMS
# 오차가 가장 작은 index를 찾음 - best centroids
cost = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)),2)
best_centroids = tf.argmin(cost, 0)


# 중심값(centroid) 갱신
# equal : best_centroid가 K개의 기준 중 어느것에 해당하는지 확인
# where : 특정 기준에 해당하는 data의 좌표를 생성하고
# reshape : tensor화 한다
# gather : 특정 기준에 해당하는 data만 모은다
# reduce_mean : 특정 기준의 data의 평균을 구한다
# update_centroides : 이제 중심값(centroid) 를 새로운 평균으로 갱신한다.
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(best_centroids, c)),[1,-1])), reduction_indices=[1]) for c in range(K)])
update_centroides = tf.assign(centroids, means)

# Tensorflow 세션 실행
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 훈련 사이클
for step in range(1000):
    _ , centroid_values, assignment_values = sess.run([update_centroides, centroids, best_centroids])
    print("Epoch : ",step)
print("학습 완료")

# 원래 데이터가 어떻게 라벨링 되었는지 출력
for i in range(len(assignment_values)):
    if assignment_values[i] == 0:
        plt.plot(data[i][0],data[i][1], 'bo')
    elif assignment_values[i] == 1:
        plt.plot(data[i][0], data[i][1], 'ro')
    elif assignment_values[i] == 2:
        plt.plot(data[i][0], data[i][1], 'mo')
    else:
        plt.plot(data[i][0], data[i][1], 'go')

# 한번 그려봅니다
print("plot 완료")
plt.show()


