import pandas
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from Face_Crop import *
from Build_CNN import *

train_data_x = pandas.read_csv('training\\train_image_data.csv')
train_data_x = train_data_x.values
train_data_y = pandas.read_csv('training\\training.csv')
test_data_x = pandas.read_csv('test\\test_image_data.csv')
test_data_x = test_data_x.values
test_data_y = pandas.read_csv('test\\test.csv')

train_data_y = train_data_y.loc[:, ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y']]

# face detection
height = width = numpy.sqrt(train_data_x.shape[1])
print(train_data_x.shape[1], height, width)
face_data_x = train_data_x
face_data_y = train_data_y.values
face_x = tf.placeholder(shape=[None, train_data_x.shape[1]], dtype=tf.float32)
face_y = tf.placeholder(shape=[None, train_data_y.shape[1]], dtype=tf.float32)

face_cnn = build_CNN()
# (7049, 96, 96, 1) > (7049, 48, 48, 16)
image = tf.reshape(face_x, [-1, 96, 96, 1])
w_conv1 = face_cnn.set_weight(shape=[4, 4, 1, 16])
b_conv1 = face_cnn.set_bias(shape=[16])
layer1 = face_cnn.add_conv_layer(image, w_conv1, stride=[1, 1, 1, 1], padding='SAME') + b_conv1
layer1_relu = tf.nn.relu(layer1)
layer1_relu_pool = face_cnn.add_pooling(layer1_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 48, 48, 16) > (7049, 24, 24, 32)
w_conv2 = face_cnn.set_weight(shape=[4, 4, 16, 32])
b_conv2 = face_cnn.set_bias(shape=[32])
layer2 = face_cnn.add_conv_layer(layer1_relu_pool, w_conv2, stride=[1, 1, 1, 1], padding='SAME') + b_conv2
layer2_relu = tf.nn.relu(layer2)
layer2_relu_pool = face_cnn.add_pooling(layer2_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 24, 24, 32) > (7049, 12, 12, 64)
w_conv3 = face_cnn.set_weight(shape=[4, 4, 32, 64])
b_conv3 = face_cnn.set_weight(shape=[64])
layer3 = face_cnn.add_conv_layer(layer2_relu_pool, w_conv3, stride=[1, 1, 1, 1], padding='SAME') + b_conv3
layer3_relu = tf.nn.relu(layer3)
layer3_relu_pool = face_cnn.add_pooling(layer3_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 12, 12, 64) > (7049, 128)
w_fc1 = face_cnn.set_weight(shape=[12 * 12 * 64, 128])
b_fc1 = face_cnn.set_bias(shape=[128])
layer3_relu_pool = tf.reshape(layer3_relu_pool, [-1, 12 * 12 * 64])
layer4 = tf.matmul(layer3_relu_pool, w_fc1) + b_fc1
layer4_relu = tf.nn.relu(layer4)
layer4_relu_drop = tf.nn.dropout(layer4_relu, keep_prob=0.5)

# (7049, 128) > (7049, 10)
w_fc2 = face_cnn.set_weight(shape=[128, 10])
b_fc2 = face_cnn.set_bias(shape=[10])
layer5 = tf.matmul(layer4_relu_drop, w_fc2) + b_fc2

cost = tf.reduce_mean(tf.reduce_sum(tf.square(layer5 - face_y) / 96, 1))
train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
# train_image_size = 7048
# train_image = train_data_x[0: train_image_size, :]
# train_y = train_data_y[0: train_image_size, :]
# val_image = train_data_x[train_image_size:, :]
# val_y = train_data_y[train_image_size:, :]
per_train_num = train_data_x.shape[0] // 100
init = tf.global_variables_initializer()
saver = tf.train.Saver()
face_coor = numpy.zeros((train_data_x.shape[0], train_data_y.shape[1]))
with tf.Session() as sess:
    sess.run(init)
    cost_list = []
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = train_data_x[start:, :]
                input_y = train_data_y[start:, :]
                feed_dict = {face_x: input_x, face_y: input_y}
                face_coor[start:, :] = sess.run(layer5, feed_dict=feed_dict)
            else:
                input_x = train_data_x[start: start + per_train_num, :]
                input_y = train_data_y[start: start + per_train_num, :]
                feed_dict = {face_x: input_x, face_y: input_y}
                face_coor[start: start + per_train_num, :] = sess.run(layer5, feed_dict=feed_dict)
            # input_x = train_data_x
            # input_y = train_data_y
            sess.run(train_op, feed_dict=feed_dict)
            cost_list.append(sess.run(cost, feed_dict=feed_dict))
    save_path = saver.save(sess, 'save/face_model_level1.ckpt')
    plt.plot(range(len(cost_list)), cost_list)
    plt.title('face_train_cost')
    plt.show()
sess.close()


face_crop = face_crop_level1(train_data_x)
# eye nose detection
en_data_x, en_top_left = face_crop.eye_nose_crop()
en_data_x = numpy.array(en_data_x)
en_data_y = train_data_y.loc[:, ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y']]
en_data_y = en_data_y - en_top_left
en_data_y = numpy.array(en_data_y)
en_x = tf.placeholder(shape=[None, en_data_x.shape[1], en_data_x.shape[2], 1], dtype=tf.float32)
en_y = tf.placeholder(shape=[None, 6], dtype=tf.float32)

en_cnn = build_CNN()
# (7049, 58, 96, 1) > (7049, 29, 48, 16)
en_w_conv1 = en_cnn.set_weight(shape=[4, 4, 1, 16])
en_b_conv1 = en_cnn.set_bias(shape=[16])
en_layer1 = en_cnn.add_conv_layer(en_x, en_w_conv1, stride=[1, 1, 1, 1], padding='SAME') + en_b_conv1
en_layer1_relu = tf.nn.relu(en_layer1)
en_layer1_relu_pool = en_cnn.add_pooling(en_layer1_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 29, 48, 16) > (7049, 15, 24, 32)
en_w_conv2 = en_cnn.set_weight(shape=[4, 4, 16, 32])
en_b_conv2 = en_cnn.set_bias(shape=[32])
en_layer2 = en_cnn.add_conv_layer(en_layer1_relu_pool, en_w_conv2, stride=[1, 1, 1, 1], padding='SAME') + en_b_conv2
en_layer2_relu = tf.nn.relu(en_layer2)
en_layer2_relu_pool = en_cnn.add_pooling(en_layer2_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 15, 24, 32) > (7049, 8, 12, 64)
en_w_conv3 = en_cnn.set_weight(shape=[4, 4, 32, 64])
en_b_conv3 = en_cnn.set_bias(shape=[64])
en_layer3 = en_cnn.add_conv_layer(en_layer2_relu_pool, en_w_conv3, stride=[1, 1, 1, 1], padding='SAME') + en_b_conv3
en_layer3_relu = tf.nn.relu(en_layer3)
en_layer3_relu_pool = en_cnn.add_pooling(en_layer3_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 8, 12, 64) > (7049, 128)
en_w_fc1 = en_cnn.set_weight(shape=[8 * 12 * 64, 128])
en_b_fc1 = en_cnn.set_bias(shape=[128])
en_layer3_relu_pool = tf.reshape(en_layer3_relu_pool, [-1, 8 * 12 * 64])
en_layer4 = tf.matmul(en_layer3_relu_pool, en_w_fc1) + en_b_fc1
en_layer4_relu = tf.nn.relu(en_layer4)
en_layer4_drop = tf.nn.dropout(en_layer4_relu, keep_prob=0.5)

# (7049, 128) > (7049, 6)
en_w_fc2 = en_cnn.set_weight(shape=[128, 6])
en_b_fc2 = en_cnn.set_bias(shape=[6])
en_layer5 = tf.matmul(en_layer4_drop, en_w_fc2) + en_b_fc2

en_cost = tf.reduce_mean(tf.reduce_sum(tf.square(en_layer5 - en_y) / 96, 1))
en_train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(en_cost)
per_train_num = en_data_x.shape[0] // 100
init = tf.global_variables_initializer()
saver = tf.train.Saver()
en_coor = numpy.zeros((en_data_x.shape[0], en_data_y.shape[1]))
with tf.Session() as sess:
    sess.run(init)
    en_cost_list = []
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = en_data_x[start:, :, :, :]
                input_y = en_data_y[start:, :]
                feed_dict = {en_x: input_x, en_y: input_y}
                en_coor[start:, :] = sess.run(en_layer5, feed_dict=feed_dict)
            else:
                input_x = en_data_x[start: start + per_train_num, :, :, :]
                input_y = en_data_y[start: start + per_train_num, :]
                feed_dict = {en_x: input_x, en_y: input_y}
                en_coor[start: start + per_train_num, :] = sess.run(en_layer5, feed_dict=feed_dict)
            sess.run(en_train_op, feed_dict=feed_dict)
            en_cost_list.append(sess.run(en_cost, feed_dict=feed_dict))
    en_coor = en_coor + en_top_left
    save_path = saver.save(sess, 'save/en_model_level1.ckpt')
    plt.plot(range(len(en_cost_list)), en_cost_list)
    plt.title('en_train_cost')
    plt.show()
sess.close()


# mouth nose detection
nm_data_x, nm_top_left = face_crop.nose_mouth_crop()
nm_data_x = numpy.array(nm_data_x)
nm_data_y = train_data_y.loc[:, ['nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y']]
nm_data_y = nm_data_y - nm_top_left
nm_data_y = numpy.array(nm_data_y)
nm_x = tf.placeholder(shape=[None, nm_data_x.shape[1], nm_data_x.shape[2], 1], dtype=tf.float32)
nm_y = tf.placeholder(shape=[None, 6], dtype=tf.float32)

nm_cnn = build_CNN()
# (7049, 48, 96, 1) > (7049, 24, 48, 16)
nm_w_conv1 = nm_cnn.set_weight(shape=[4, 4, 1, 16])
nm_b_conv1 = nm_cnn.set_bias(shape=[16])
nm_layer1 = nm_cnn.add_conv_layer(nm_x, nm_w_conv1, stride=[1, 1, 1, 1], padding='SAME') + nm_b_conv1
nm_layer1_relu = tf.nn.relu(nm_layer1)
nm_layer1_relu_pool = nm_cnn.add_pooling(nm_layer1_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 24, 48, 16) > (7049, 12, 24, 32)
nm_w_conv2 = nm_cnn.set_weight(shape=[4, 4, 16, 32])
nm_b_conv2 = nm_cnn.set_bias(shape=[32])
nm_layer2 = nm_cnn.add_conv_layer(nm_layer1_relu_pool, nm_w_conv2, stride=[1, 1, 1, 1], padding='SAME') + nm_b_conv2
nm_layer2_relu = tf.nn.relu(nm_layer2)
nm_layer2_relu_pool = nm_cnn.add_pooling(nm_layer2_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 12, 24, 32) > (7049, 6, 12, 64)
nm_w_conv3 = nm_cnn.set_weight(shape=[4, 4, 32, 64])
nm_b_conv3 = nm_cnn.set_bias(shape=[64])
nm_layer3 = nm_cnn.add_conv_layer(nm_layer2_relu_pool, nm_w_conv3, stride=[1, 1, 1, 1], padding='SAME') + nm_b_conv3
nm_layer3_relu = tf.nn.relu(nm_layer3)
nm_layer3_relu_pool = nm_cnn.add_pooling(nm_layer3_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

# (7049, 6, 12, 64) > (7049, 128)
nm_w_fc1 = nm_cnn.set_weight(shape=[6 * 12 * 64, 128])
nm_b_fc1 = nm_cnn.set_bias(shape=[128])
nm_layer3_relu_pool = tf.reshape(nm_layer3_relu_pool, [-1, 2 * 12 * 64])
nm_layer4 = tf.matmul(nm_layer3_relu_pool, nm_w_fc1) + nm_b_fc1
nm_layer4_relu = tf.nn.relu(nm_layer4)
nm_layer4_drop = tf.nn.dropout(nm_layer4_relu, keep_prob=0.5)

# (7049, 128) > (7049, 6)
nm_w_fc2 = nm_cnn.set_weight(shape=[128, 6])
nm_b_fc2 = nm_cnn.set_bias(shape=[6])
nm_layer5 = tf.matmul(nm_layer4_drop, nm_w_fc2) + nm_b_fc2

nm_cost = tf.reduce_mean(tf.reduce_sum(tf.square(nm_layer5 - nm_y) / 96, 1))
nm_train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(nm_cost)
per_train_num = nm_data_x.shape[0] // 100
init = tf.global_variables_initializer()
saver = tf.train.Saver()
nm_coor = numpy.zeros((nm_data_x.shape[0], nm_data_y.shape[1]))
with tf.Session() as sess:
    sess.run(init)
    nm_cost_list = []
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = nm_data_x[start:, :, :, :]
                input_y = nm_data_y[start:, :]
                feed_dict = {nm_x: input_x, nm_y: input_y}
                nm_coor[start:, :] = sess.run(nm_layer5, feed_dict=feed_dict)
            else:
                input_x = nm_data_x[start: start + per_train_num, :, :, :]
                input_y = nm_data_y[start: start + per_train_num, :]
                feed_dict = {nm_x: input_x, nm_y: input_y}
                nm_coor[start: start + per_train_num, :] = sess.run(nm_layer5, feed_dict=feed_dict)
            sess.run(nm_train_op, feed_dict=feed_dict)
            nm_cost_list.append(sess.run(nm_cost, feed_dict=feed_dict))
    nm_coor = nm_coor + nm_top_left
    save_path = saver.save(sess, 'save/nm_model_level1.ckpt')
    plt.plot(range(len(nm_cost_list)), nm_cost_list)
    plt.title('nm_train_cost')
    plt.show()
sess.close()

coor_data = numpy.array((train_data_y.shape[0], train_data_y.shape[1]))
coor_data[:, 0] = (face_coor[:, 0] + en_coor[:, 0]) / 2
coor_data[:, 1] = (face_coor[:, 1] + en_coor[:, 1]) / 2
coor_data[:, 2] = (face_coor[:, 2] + en_coor[:, 2]) / 2
coor_data[:, 3] = (face_coor[:, 3] + en_coor[:, 3]) / 2
coor_data[:, 4] = (face_coor[:, 4] + en_coor[:, 4] + nm_coor[:, 1]) / 3
coor_data[:, 5] = (face_coor[:, 5] + en_coor[:, 5] + nm_coor[:, 2]) / 3
coor_data[:, 6] = (face_coor[:, 6] + nm_coor[:, 3]) / 2
coor_data[:, 7] = (face_coor[:, 7] + nm_coor[:, 4]) / 2
coor_data[:, 8] = (face_coor[:, 8] + nm_coor[:, 5]) / 2
coor_data[:, 9] = (face_coor[:, 9] + nm_coor[:, 6]) / 2
coor_data = pandas.DataFrame(coor_data)
coor_data.to_csv('train_coor_data.csv')
