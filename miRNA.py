
# coding: utf-8

# In[1]:

'''
A Multilayer Perceptron implementation for RNA prediction.
2 hidden layers with no regulation,using softmax function with cross-entropy.

'''


# In[2]:

import numpy as np
import random
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

learning_rate = 1e-2
REGULARIZATION_RATE = 0.0001


# In[3]:

positive_training_data = "positive_training_dataset.txt"
negative_training_data = "negative_training_dataset.txt"
positive_testing_data = "positive_testing_dataset.txt"
negative_testing_data = "negative_testing_dataset.txt"
sess = tf.InteractiveSession()


# In[4]:

def RNA_to_feature(str_rna, str_structure):

    # 接受两个字符串,第一个字符串是基因序列,第二个字符串是对应的二级结构

    energy = re.findall(r'-{0,1}\d{1,2}.\d{1,2}', str_structure)[0]  # 利用正则表达式提取自由能

    rna, structure = '', ''

    for i in range(len(str_rna)):

        if str_rna[i] is not 'A' and str_rna[i] is not 'C' and str_rna[i] is not 'G' and str_rna[i] is not 'U':
            continue

        rna += str_rna[i]
        structure += str_structure[i]

    structure2int = [0 for i in range(len(structure))]

    for i in range(len(structure)):
        if structure[i] is '(' or ')':
            structure2int[i]= 1
        if structure[i] is '.':
            structure2int[i]= 0

    rna_dic = {'A000': 0, 'A001': 0, 'A010': 0, 'A011': 0, 'A100': 0,'A101': 0, 'A110': 0, 'A111': 0,
          'C000': 0, 'C001': 0, 'C010': 0, 'C011': 0, 'C100': 0,'C101': 0, 'C110': 0, 'C111': 0,
          'G000': 0, 'G001': 0, 'G010': 0, 'G011': 0, 'G100': 0,'G101': 0, 'G110': 0, 'G111': 0,
          'U000': 0, 'U001': 0, 'U010': 0, 'U011': 0, 'U100': 0,'U101': 0, 'U110': 0, 'U111': 0}
    rna_index = ['A000', 'A001', 'A010', 'A011', 'A100','A101', 'A110', 'A111',
        'C000', 'C001', 'C010', 'C011', 'C100','C101', 'C110', 'C111',
        'G000', 'G001', 'G010', 'G011', 'G100','G101', 'G110', 'G111',
        'U000', 'U001', 'U010', 'U011', 'U100','U101', 'U110', 'U111']
    for i in range(len(rna)-2):
        str2all = rna[i]+ __builtins__.str(structure2int[i])+ __builtins__.str(structure2int[i+1])+ __builtins__.str(structure2int[i+2])
        rna_dic[str2all] += 1
    formalInput = [0 for i in range(33)]
    count = 0;
    for i in rna_index:
        formalInput[count] =  rna_dic[i]
        count += 1
    formalInput[32] = float(energy)

    return formalInput


# In[5]:

def input_form_x(data_path):


    with open(data_path) as data:
        form_x = []
        fp = data.readlines()
        for i in range(0, len(fp), 2):

            form_x.append(RNA_to_feature(fp[i], fp[i + 1]))

    return form_x


# In[6]:

def data_process():
    X_train_correct = input_form_x(positive_training_data)  # 获取数据
#     Y_correct = [1 for i in range(len(X_correct))]  # target initialization
    Y_train_correct = []
    for i in range(len(X_train_correct)):
        Y_train_correct.append([1,0])

    X_train_wrong = input_form_x(negative_training_data)
#     Y_wrong = [0 for i in range(len(X_wrong))]
    Y_train_wrong = []
    for i in range(len(X_train_wrong)):
        Y_train_wrong.append([0,1])

    X_test_correct = input_form_x(positive_testing_data)  # 获取数据
#     Y_correct = [1 for i in range(len(X_correct))]  # target initialization
    Y_test_correct = []
    for i in range(len(X_test_correct)):
        Y_test_correct.append([1,0])

    X_test_wrong = input_form_x(negative_testing_data)
#     Y_wrong = [0 for i in range(len(X_wrong))]
    Y_test_wrong = []
    for i in range(len(X_test_wrong)):
        Y_test_wrong.append([0,1])



    return X_train_correct,Y_train_correct,X_train_wrong ,Y_train_wrong ,X_test_correct ,Y_test_correct ,X_test_wrong ,Y_test_wrong


# In[7]:

# INPUT NODES FOR PASSING DATA INTO THE GRAPH
x = tf.placeholder(tf.float32, [None, 33])
y = tf.placeholder(tf.float32, [None, 2])
#keep_prob = tf.placeholder("float")
# y_ = tf.placeholder(tf.float32 , [None , 1])


# In[8]:

# Create model

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     layer_2 = tf.nn.relu(out_layer)
    return out_layer


# In[9]:

weights = {
    'h1': tf.Variable(tf.random_normal([33, 10])),
    'h2': tf.Variable(tf.random_normal([10, 20])),
    'out': tf.Variable(tf.random_normal([20, 2]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([10])),
    'b2': tf.Variable(tf.random_normal([20])),
    'out': tf.Variable(tf.random_normal([2]))
}

pred = multilayer_perceptron(x, weights, biases)

logits = tf.nn.softmax(pred)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# In[10]:

X_train_correct, Y_train_correct, X_train_wrong, Y_train_wrong, X_test_correct, Y_test_correct ,X_test_wrong, Y_test_wrong = data_process()


# In[11]:

print len(X_train_correct)
print len(Y_train_correct)
print len(X_train_wrong)
print len(Y_train_wrong)
print len(X_test_correct)
print len(Y_test_correct)
print len(X_test_wrong)
print len(Y_test_wrong)


# In[12]:

x_train = X_train_correct + X_train_wrong
x_test = X_test_correct + X_test_wrong
y_train = Y_train_correct + Y_train_wrong
y_test = Y_test_correct + Y_test_wrong


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x_train + x_test, y_train + y_test, test_size=0.15,
                                                        random_state=42)  # 训练集和测试集的随机分割
print len(x_train)
print len(y_train)
print len(x_test)
print len(y_test)
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[18]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)


#     Training cycle
    for epoch in range(8000):
        batch_index = random.randint(0, 1000)

        batch_data = x_train[batch_index:batch_index + 32]
        batch_labels = y_train[batch_index:batch_index + 32]

        sess.run([optimizer, cost], feed_dict={x: batch_data, y: batch_labels})
        if epoch%300 == 0:
            print epoch
            print "Accuracy:", accuracy.eval({x:x_test, y:y_test})



    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Total Accuracy:", accuracy.eval({x:x_test, y:y_test})
