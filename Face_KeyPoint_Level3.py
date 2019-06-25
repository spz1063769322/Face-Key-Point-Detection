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

coor_data_level2 = pandas.read_csv('coor_data_level2.csv')
coor_data_level2 = coor_data_level2.values

face_crop_level3 = face_crop_level3(train_data_x, coor_data_level2)
left_eye_data, left_eye_top_left = face_crop_level3.left_eye_crop()
left_eye_data = numpy.array(left_eye_data)
right_eye_data, right_eye_top_left = face_crop_level3.right_eye_crop()
right_eye_data = numpy.array(right_eye_data)
nose_data, nose_top_left = face_crop_level3.nose_crop()
nose_data = numpy.array(nose_data)
left_mouth_data, left_mouth_top_left = face_crop_level3.left_mouth_crop()
left_mouth_data = numpy.array(left_mouth_data)
right_mouth_data, right_mouth_top_left = face_crop_level3.right_mouth_crop()
right_mouth_data = numpy.array(right_mouth_data)


# left eye NN1
left_eye_coor = train_data_y.loc[:, ['left_eye_center_x', 'left_eye_center_y']]
left_eye_coor = left_eye_coor.values
left_eye_coor = left_eye_coor - left_eye_top_left
x = tf.placeholder(shape=[None, left_eye_data.shape[1], left_eye_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, left_eye_coor.shape[1]], dtype=tf.float32)
left_eye_train_op, left_eye_layer4, left_eye_cost = build_keypoint_detection(left_eye_data, left_eye_coor, x, y)
per_train_num = left_eye_coor.shape[0] // 100
saver = tf.train.Saver()
left_eye_cost_list = []
left_eye_coor_level3 = numpy.zeros((left_eye_coor.shape[0], left_eye_coor.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = left_eye_data[start:, :, :, :]
                input_y = left_eye_coor[start:, :]
                feed_dict = {x: input_x, y: input_y}
                left_eye_coor_level3[start:, :] = sess.run(left_eye_layer4, feed_dict=feed_dict)
            else:
                input_x = left_eye_data[start: start + per_train_num, :, :, :]
                input_y = left_eye_coor[start: start + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                left_eye_coor_level3[start: start + per_train_num, :] = sess.run(left_eye_layer4, feed_dict=feed_dict)
            _, cost = sess.run([left_eye_train_op, left_eye_cost], feed_dict=feed_dict)
            left_eye_cost_list.append(cost)
    left_eye_coor_level3 = left_eye_coor_level3 + left_eye_top_left
    save_path = saver.save(sess, 'save/left_eye_model_level3_1')
    plt.plot(range(len(left_eye_cost_list)), left_eye_cost_list)
    plt.title('left_eye_cost_list_level3_1')
    plt.show()
sess.close()


# right eye NN1
right_eye_coor = train_data_y.loc[:, ['right_eye_center_x', 'right_eye_center_y']]
right_eye_coor = right_eye_coor.values
right_eye_coor = right_eye_coor - right_eye_top_left
x = tf.placeholder(shape=[None, right_eye_data.shape[1], right_eye_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, right_eye_coor.shape[1]], dtype=tf.float32)
right_eye_train_op, right_eye_layer4, right_eye_cost = build_keypoint_detection(right_eye_data, right_eye_coor, x, y)
per_train_num = right_eye_coor.shape[0] // 100
saver = tf.train.Saver()
right_eye_cost_list = []
right_eye_coor_level3 = numpy.zeros((right_eye_coor.shape[0], right_eye_coor.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = right_eye_data[start:, :, :, :]
                input_y = right_eye_coor[start:, :]
                feed_dict = {x: input_x, y: input_y}
                right_eye_coor_level3[start:, :] = sess.run(right_eye_layer4, feed_dict=feed_dict)
            else:
                input_x = right_eye_data[start: start + per_train_num, :, :, :]
                input_y = right_eye_coor[start: start + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                right_eye_coor_level3[start: start + per_train_num, :] = sess.run(right_eye_layer4, feed_dict=feed_dict)
            _, cost = sess.run([right_eye_train_op, right_eye_cost], feed_dict=feed_dict)
            right_eye_cost_list.append(cost)
    right_eye_coor_level3 = right_eye_coor_level3 + right_eye_top_left
    save_path = saver.save(sess, 'save/right_eye_model_level3_1')
    plt.plot(range(len(right_eye_cost_list)), right_eye_cost_list)
    plt.title('right_eye_cost_list_level3_1')
    plt.show()
sess.close()


# nose NN1
nose_coor = train_data_y.loc[:, ['nose_tip_x', 'nose_tip_y']]
nose_coor = nose_coor.values
nose_coor = nose_coor - nose_top_left
x = tf.placeholder(shape=[None, nose_data.shape[1], nose_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, nose_coor.shape[1]], dtype=tf.float32)
nose_train_op, nose_layer4, nose_cost = build_keypoint_detection(nose_data, nose_coor, x, y)
per_train_num = nose_coor.shape[0] // 100
saver = tf.train.Saver()
nose_cost_list = []
nose_coor_level3 = numpy.zeros((nose_coor.shape[0], nose_coor.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = nose_data[start:, :, :, :]
                input_y = nose_coor[start:, :]
                feed_dict = {x: input_x, y: input_y}
                nose_coor_level3[start:, :] = sess.run(nose_layer4, feed_dict=feed_dict)
            else:
                input_x = nose_data[start: start + per_train_num, :, :, :]
                input_y = nose_coor[start: start + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                nose_coor_level3[start: start + per_train_num, :] = sess.run(nose_layer4, feed_dict=feed_dict)
            _, cost = sess.run([nose_train_op, nose_cost], feed_dict=feed_dict)
            nose_cost_list.append(cost)
    nose_coor_level3 = nose_coor_level3 + nose_top_left
    save_path = saver.save(sess, 'save/nose_model_level3_1')
    plt.plot(range(len(nose_cost_list)), nose_cost_list)
    plt.title('nose_cost_list_level3_1')
    plt.show()
sess.close()


# left mouth NN1
left_mouth_coor = train_data_y.loc[:, ['mouth_left_corner_x', 'mouth_left_corner_y']]
left_mouth_coor = left_mouth_coor.values
left_mouth_coor = left_mouth_coor - left_mouth_top_left
x = tf.placeholder(shape=[None, left_mouth_data.shape[1], left_mouth_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, left_mouth_coor.shape[1]], dtype=tf.float32)
left_mouth_train_op, left_mouth_layer4, left_mouth_cost = build_keypoint_detection(left_mouth_data, left_mouth_coor, x, y)
per_train_num = left_mouth_coor.shape[0] // 100
saver = tf.train.Saver()
left_mouth_cost_list = []
left_mouth_coor_level3 = numpy.zeros((left_mouth_coor.shape[0], left_mouth_coor.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = left_mouth_data[start:, :, :, :]
                input_y = left_mouth_coor[start:, :]
                feed_dict = {x: input_x, y: input_y}
                left_mouth_coor_level3[start:, :] = sess.run(left_mouth_layer4, feed_dict=feed_dict)
            else:
                input_x = left_mouth_data[start: start + per_train_num, :, :, :]
                input_y = left_mouth_coor[start: start + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                left_mouth_coor_level3[start: start + per_train_num, :] = sess.run(left_mouth_layer4, feed_dict=feed_dict)
            _, cost = sess.run([left_mouth_train_op, left_mouth_cost], feed_dict=feed_dict)
            left_mouth_cost_list.append(cost)
    left_mouth_coor_level3 = left_eye_coor_level3 + left_mouth_top_left
    save_path = saver.save(sess, 'save/left_mouth_model_level3_1')
    plt.plot(range(len(left_mouth_cost_list)), left_mouth_cost_list)
    plt.title('left_mouth_cost_list_level3_1')
    plt.show()
sess.close()


# right mouth NN1
right_mouth_coor = train_data_y.loc[:, ['mouth_right_corner_x', 'mouth_right_corner_y']]
right_mouth_coor = right_mouth_coor.values
right_mouth_coor = right_mouth_coor - right_mouth_top_left
x = tf.placeholder(shape=[None, right_mouth_data.shape[1], right_mouth_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, right_mouth_coor.shape[1]], dtype=tf.float32)
right_mouth_train_op, right_mouth_layer4, right_mouth_cost = build_keypoint_detection(right_mouth_data, right_mouth_coor, x, y)
per_train_num = right_mouth_coor.shape[0] // 100
saver = tf.train.Saver()
right_mouth_cost_list = []
right_mouth_coor_level3 = numpy.zeros((right_mouth_coor.shape[0], right_mouth_coor.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            if(j == 99):
                input_x = right_mouth_data[start:, :, :, :]
                input_y = right_mouth_coor[start:, :]
                feed_dict = {x: input_x, y: input_y}
                right_mouth_coor_level3[start:, :] = sess.run(right_mouth_layer4, feed_dict=feed_dict)
            else:
                input_x = right_mouth_data[start: start + per_train_num, :, :, :]
                input_y = right_mouth_coor[start: start + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                right_mouth_coor_level3[start: start + per_train_num, :] = sess.run(right_mouth_layer4, feed_dict=feed_dict)
            _, cost = sess.run([right_mouth_train_op, right_mouth_cost], feed_dict=feed_dict)
            right_mouth_cost_list.append(cost)
    right_mouth_coor_level3 = right_mouth_coor_level3 + right_mouth_top_left
    save_path = saver.save(sess, 'save/right_mouth_model_level3_1')
    plt.plot(range(len(right_mouth_cost_list)), right_mouth_cost_list)
    plt.title('right_mouth_cost_list_level3_1')
    plt.show()
sess.close()


# left eye NN2
left_eye_coor_1 = train_data_y.loc[:, ['left_eye_center_x', 'left_eye_center_y']]
left_eye_coor_1 = left_eye_coor_1.values
left_eye_coor_1 = left_eye_coor_1 - left_eye_top_left
x = tf.placeholder(shape=[None, left_eye_data.shape[1], left_eye_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, left_eye_coor_1.shape[1]], dtype=tf.float32)
left_eye_train_op_1, left_eye_layer4_1, left_eye_cost_1 = build_keypoint_detection(left_eye_data, left_eye_coor_1, x, y)
per_train_num = left_eye_coor_1.shape[0] // 100
saver = tf.train.Saver()
left_eye_cost_list_1 = []
left_eye_coor_level3_1 = numpy.zeros((left_eye_coor_1.shape[0], left_eye_coor_1.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            index = train_data_x.shape[0] - start
            if(j == 99):
                input_x = left_eye_data[: index, :, :, :]
                input_y = left_eye_coor_1[: index, :]
                feed_dict = {x: input_x, y: input_y}
                left_eye_coor_level3_1[: index, :] = sess.run(left_eye_layer4_1, feed_dict=feed_dict)
            else:
                input_x = left_eye_data[index: index + per_train_num, :, :, :]
                input_y = left_eye_coor_1[index: index + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                left_eye_coor_level3_1[index: index + per_train_num, :] = sess.run(left_eye_layer4_1, feed_dict=feed_dict)
            _, cost = sess.run([left_eye_train_op_1, left_eye_cost_1], feed_dict=feed_dict)
            left_eye_cost_list_1.append(cost)
    left_eye_coor_level3_1 = left_eye_coor_level3_1 + left_eye_top_left
    save_path = saver.save(sess, 'save/left_eye_model_level3_1')
    plt.plot(range(len(left_eye_cost_list_1)), left_eye_cost_list_1)
    plt.title('left_eye_cost_list_level3_1')
    plt.show()
sess.close()


# right eye NN2
right_eye_coor_1 = train_data_y.loc[:, ['right_eye_center_x', 'right_eye_center_y']]
right_eye_coor_1 = right_eye_coor_1.values
right_eye_coor_1 = right_eye_coor_1 - right_eye_top_left
x = tf.placeholder(shape=[None, right_eye_data.shape[1], right_eye_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, right_eye_coor_1.shape[1]], dtype=tf.float32)
right_eye_train_op_1, right_eye_layer4_1, right_eye_cost_1 = build_keypoint_detection(right_eye_data, right_eye_coor_1, x, y)
per_train_num = right_eye_coor_1.shape[0] // 100
saver = tf.train.Saver()
right_eye_cost_list_1 = []
right_eye_coor_level3_1 = numpy.zeros((right_eye_coor_1.shape[0], right_eye_coor_1.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            index = train_data_x.shape[0] - start
            if(j == 99):
                input_x = right_eye_data[: index, :, :, :]
                input_y = right_eye_coor_1[: index, :]
                feed_dict = {x: input_x, y: input_y}
                right_eye_coor_level3_1[: index, :] = sess.run(right_eye_layer4_1, feed_dict=feed_dict)
            else:
                input_x = right_eye_data[index: index + per_train_num, :, :, :]
                input_y = right_eye_coor_1[index: index + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                right_eye_coor_level3_1[index: index + per_train_num, :] = sess.run(right_eye_layer4_1, feed_dict=feed_dict)
            _, cost = sess.run([right_eye_train_op_1, right_eye_cost_1], feed_dict=feed_dict)
            right_eye_cost_list_1.append(cost)
    right_eye_coor_level3_1 = right_eye_coor_level3_1 + right_eye_top_left
    save_path = saver.save(sess, 'right_eye_model_level3_2')
    plt.plot(range(len(right_eye_cost_list_1)), right_eye_cost_list_1)
    plt.title('right_eye_cost_list_level3_2')
    plt.show()
sess.close()


# nose NN2
nose_coor_1 = train_data_y.loc[:, ['nose_tip_x', 'nose_tip_y']]
nose_coor_1 = nose_coor_1.values
nose_coor_1 = nose_coor_1 - nose_top_left
x = tf.placeholder(shape=[None, nose_data.shape[1], nose_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, nose_coor_1.shape[1]], dtype=tf.float32)
nose_train_op_1, nose_layer4_1, nose_cost_1 = build_keypoint_detection(nose_data, nose_coor_1, x, y)
per_train_num = nose_coor_1.shape[0] // 100
saver = tf.train.Saver()
nose_cost_list_1 = []
nose_coor_level3_1 = numpy.zeros((nose_coor_1.shape[0], nose_coor_1.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            index = train_data_x.shape[0] - start
            if(j == 99):
                input_x = nose_data[: index, :, :, :]
                input_y = nose_coor_1[: index, :]
                feed_dict = {x: input_x, y: input_y}
                nose_coor_level3_1[: index, :] = sess.run(nose_layer4_1, feed_dict=feed_dict)
            else:
                input_x = nose_data[index: index + per_train_num, :, :, :]
                input_y = nose_coor_1[index: index + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                nose_coor_level3_1[index: index + per_train_num, :] = sess.run(nose_layer4_1, feed_dict=feed_dict)
            _, cost = sess.run([nose_train_op_1, nose_cost_1], feed_dict=feed_dict)
            nose_cost_list_1.append(cost)
    nose_coor_level3_1 =nose_coor_level3_1 + nose_top_left
    sace_path = saver.save(sess, 'save/nose_model_level3_2')
    plt.plot(range(len(nose_cost_list_1)), nose_cost_list_1)
    plt.title('nose_cost_list_level3_2')
    plt.show()
sess.close()


# left mouth NN2
left_mouth_coor_1 = train_data_y.loc[:, ['mouth_left_corner_x', 'mouth_left_corner_y']]
left_mouth_coor_1 = left_mouth_coor_1.values
left_mouth_coor_1 = left_mouth_coor_1 - left_mouth_top_left
x = tf.placeholder(shape=[None, left_mouth_data.shape[1], left_mouth_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, left_mouth_coor_1.shape[1]], dtype=tf.float32)
left_mouth_train_op_1, left_mouth_layer4_1, left_mouth_cost_1 = build_keypoint_detection(left_mouth_data, left_mouth_coor_1, x, y)
per_train_num = left_mouth_coor_1.shape[0] // 100
saver = tf.train.Saver()
left_mouth_cost_list_1 = []
left_mouth_coor_level3_1 = numpy.zeros((left_mouth_coor_1.shape[0], left_mouth_coor_1.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            index = train_data_x.shape[0] - start
            if(j == 99):
                input_x = left_mouth_data[: index, :, :, :]
                input_y = left_mouth_coor_1[: index, :]
                feed_dict = {x: input_x, y: input_y}
                left_mouth_coor_level3_1[: index, :] = sess.run(left_mouth_layer4_1, feed_dict=feed_dict)
            else:
                input_x = left_mouth_data[index: index + per_train_num, :, :, :]
                input_y = left_mouth_coor_1[index: index + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                left_mouth_coor_level3_1[index: index + per_train_num, :] = sess.run(left_mouth_layer4_1, feed_dict=feed_dict)
            _, cost = sess.run([left_mouth_train_op_1, left_mouth_cost_1], feed_dict=feed_dict)
            left_mouth_cost_list_1.append(cost)
    left_mouth_coor_level3_1 = left_mouth_coor_level3_1 + left_mouth_top_left
    save_path = saver.save(sess, 'save/left_mouth_model_level3_2')
    plt.plot(range(len(left_mouth_cost_list_1)), left_mouth_cost_list_1)
    plt.title('left_mouth_cost_list_level3_2')
    plt.show()
sess.close()


# right mouth NN2
right_mouth_coor_1 = train_data_y.loc[:, ['mouth_right_corner_x', 'mouth_right_corner_y']]
right_mouth_coor_1 = right_mouth_coor_1.values
right_mouth_coor_1 = right_mouth_coor_1 - right_mouth_top_left
x = tf.placeholder(shape=[None, right_mouth_data.shape[1], right_mouth_data.shape[2], 1], dtype=tf.float32)
y = tf.placeholder(shape=[None, right_mouth_coor_1.shape[1]], dtype=tf.float32)
right_mouth_train_op_1, right_mouth_layer4_1, right_mouth_cost_1 = build_keypoint_detection(right_mouth_data, right_mouth_coor_1, x, y)
per_train_num = right_mouth_coor_1.shape[0] // 100
saver = tf.train.Saver()
right_mouth_cost_list_1 = []
right_mouth_coor_level3_1 = numpy.zeros((right_mouth_coor_1.shape[0], right_mouth_coor_1.shape[1]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(100):
            start = j * per_train_num
            index = train_data_x.shape[0] - start
            if(j == 99):
                input_x = right_mouth_data[: index, :, :, :]
                input_y = right_mouth_coor_1[: index, :]
                feed_dict = {x: input_x, y: input_y}
                right_mouth_coor_level3_1[: index, :] = sess.run(right_mouth_layer4_1, feed_dict=feed_dict)
            else:
                input_x = right_mouth_data[index: index + per_train_num, :, :, :]
                input_y = right_mouth_coor_1[index: index + per_train_num, :]
                feed_dict = {x: input_x, y: input_y}
                right_mouth_coor_level3_1[index: index + per_train_num, :] = sess.run(right_mouth_layer4_1, feed_dict=feed_dict)
            _, cost = sess.run([right_mouth_train_op_1, right_mouth_cost_1], feed_dict=feed_dict)
            right_mouth_cost_list_1.append(cost)
    right_mouth_coor_level3_1 = right_mouth_coor_level3_1 + right_mouth_top_left
    save_path = saver.save(sess, 'save/right_mouth_model_level3_2')
    plt.plot(range(len(right_mouth_cost_list_1)), right_mouth_cost_list_1)
    plt.title('right_mouth_cost_list_level3_2')
    plt.show()
sess.close()


coor_data_level3 = numpy.array((train_data_y.shape[0], train_data_y.shape[1]))
coor_data_level3[:, 0] = (left_eye_coor_level3[:, 0] + left_eye_coor_level3_1[:, 0]) / 2
coor_data_level3[:, 1] = (left_eye_coor_level3[:, 1] + left_eye_coor_level3_1[:, 1]) / 2
coor_data_level3[:, 2] = (right_eye_coor_level3[:, 0] + right_eye_coor_level3_1[:, 0]) / 2
coor_data_level3[:, 3] = (right_eye_coor_level3[:, 1] + right_eye_coor_level3_1[:, 1]) / 2
coor_data_level3[:, 4] = (nose_coor_level3[:, 0] + nose_coor_level3_1[:, 0]) / 2
coor_data_level3[:, 5] = (nose_coor_level3[:, 1] + nose_coor_level3_1[:, 1]) / 2
coor_data_level3[:, 6] = (left_mouth_coor_level3[:, 0] + left_mouth_coor_level3_1[:, 0]) / 2
coor_data_level3[:, 7] = (left_mouth_coor_level3[:, 1] + left_mouth_coor_level3_1[:, 1]) / 2
coor_data_level3[:, 8] = (right_mouth_coor_level3[:, 0] + right_mouth_coor_level3_1[:, 0]) / 2
coor_data_level3[:, 9] = (right_mouth_coor_level3[:, 1] + right_mouth_coor_level3_1[:, 1]) / 2