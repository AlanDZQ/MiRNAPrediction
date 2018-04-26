# coding:utf-8

import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

LEARNING_RATE = 1e-2
BATCH_SIZE = 156
INPUT_NODE = 33
OUTPUT_NODE = 2
TRAINING_STEPS = 660
REGULARIZATION_RATE = 0.0001

data_path = 'pre2.txt'
pseodu_data_path = 'psedresult2.txt'


def RNA_to_feature_list(str_rna, str_cod):
    # 接受两个字符串,第一个字符串是基因序列,第二个字符串是对应的二级结构,最终把两段序列转化为目标向量
    structure = re.findall(r'-{0,1}\d{1,2}.\d{1,2}', str_cod)[0]  # 利用正则表达式提取自由能
    rna, cod = '', ''
    for i in range(len(str_rna)):
        if str_rna[i] is not 'A' and str_rna[i] is not 'C' and str_rna[i] is not 'G' and str_rna[i] is not 'U' and \
                        str_rna[i] is not 'T':
            continue
        rna += str_rna[i]
        cod += str_cod[i]
    counter = 1
    for i in cod:
        if i == '(':
            counter += 1.0 / 2
    rna_dic = {'(': '0', '.': '1', 'A': '00', 'C': '01', 'G': '10', 'U': '11'}
    counter_dic = {}
    for i in range(32):  # initial counter_dic
        counter_dic[i] = 0
    target_list = []
    for i in range(0, len(rna) - 3):
        counter_dic[
            int(rna_dic[rna[i]] + str(reduce(lambda x, y: x + y, map(lambda x: rna_dic[x], cod[i:i + 3]))), 2)] += 1
        # 利用二进制把32种情况转换为0到31这32个数字之间,在字典里计算各种出现的次数
    for i in range(32):
        target_list.append(counter_dic[i])
    p1 = -float(structure) / 1000
    p2 = -float(structure) / (10 * counter)  # 后续的四维特征可以在这里添加
    return target_list + [float(structure)]  # 增加特征只需要改为[p1,p2,p3,p4]即可


def get_X(data_path):
    # 接受一个文件地址,读取其中的基因序列,二级结构和最小自由能,返回一个二维序列
    with open(data_path) as tt:
        tmp_lst = []
        fp = tt.readlines()
        for i in range(0, len(fp), 2):
            if len(fp[i]) > 56:
                tmp_lst.append(RNA_to_feature_list(fp[i], fp[i + 1]))
    return tmp_lst


def data_process(data_path, pseodu_data_path):
    X_correct = get_X(data_path)  # 获取数据
    Y_correct = [[1, 0] for i in range(len(X_correct))]  # target initialization

    X_pseudo = get_X(pseodu_data_path)[:len(X_correct)]
    Y_pseudo = [[0, 1] for i in range(len(X_pseudo))]

    # data normalization
    X_correct = preprocessing.normalize(X_correct).tolist()  # 把np数组转化为python的list
    X_pseudo = preprocessing.normalize(X_pseudo).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_correct + X_pseudo, Y_correct + Y_pseudo, test_size=0.15,
                                                        random_state=42)  # 训练集和测试集的随机分割

    return X_train, X_test, y_train, y_test


# INPUT NODES FOR PASSING DATA INTO THE GRAPH
x = tf.placeholder(tf.float32, [None, INPUT_NODE])
y = tf.placeholder(tf.float32, [None, OUTPUT_NODE])


# two layer neural

def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights[1]), _biases[1]))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights[2]), _biases[2]))
    return tf.matmul(layer_3, _weights[3]) + _biases[3]


weights = [
    tf.Variable(tf.random_normal([33, 70], seed=888)),
    tf.Variable(tf.random_normal([70, 30], seed=888)),
    tf.Variable(tf.random_normal([30, 10], seed=888)),
    tf.Variable(tf.random_normal([10, 2], seed=888))
]
biases = [
    tf.Variable(tf.random_normal([70], seed=888)),
    tf.Variable(tf.random_normal([30], seed=888)),
    tf.Variable(tf.random_normal([10], seed=888)),
    tf.Variable(tf.random_normal([2], seed=888))
]

pred = multilayer_perceptron(x, weights, biases)

logits = tf.nn.softmax(pred)

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularzation = regularizer(weights[0]) + regularizer(weights[1]) + regularizer(weights[2]) + regularizer(weights[3])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + regularzation

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    X_train, X_test, y_train, y_test = data_process(data_path, pseodu_data_path)

    data_len = len(X_train)

    for step in range(TRAINING_STEPS):
        batch_index = (step * BATCH_SIZE) % (data_len + 1 - BATCH_SIZE)
        batch_data = X_train[batch_index:batch_index + BATCH_SIZE]
        batch_labels = y_train[batch_index:batch_index + BATCH_SIZE]
        sess.run(optimizer, feed_dict={x: batch_data, y: batch_labels})

        if step % 100 == 0 or step == TRAINING_STEPS - 1:
            # Testing accuracy

            outputs = sess.run(logits, feed_dict={x: X_test})
            match = 0

            for i in range(len(outputs)):
                if outputs[i][0] > outputs[i][1]:
                    output = [1, 0]
                else:
                    output = [0, 1]
                if output == y_test[i]:
                    match += 1

            print match * 1.0 / len(outputs)
