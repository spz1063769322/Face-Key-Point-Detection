import tensorflow as tf

class build_CNN():

    def add_conv_layer(self, input_x, filter, stride, padding=None):
        if(padding != None):
            return tf.nn.conv2d(input_x, filter=filter, strides=stride, padding = padding)
        else:
            return tf.nn.conv2d(input_x, filter=filter, strides=stride)

    def add_pooling(self, input_x, stride, ksize, padding=None):
        if(padding != None):
            return tf.nn.max_pool(input_x, strides=stride, ksize=ksize, padding=padding)
        else:
            return tf.nn.max_pool(input_x, ksize=ksize, strides=stride)

    def set_weight(self, shape):
        init = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(init)

    def set_bias(self, shape):
        init = tf.truncated_normal(shape=shape, stddev=0.01)
        return tf.Variable(init)


def build_keypoint_detection(image_data, coor_data, x, y):
    height = image_data.shape[1]
    width = image_data.shape[2]

    nn = build_CNN()
    # (7049, height, width ,1) > (7049, height / 2, width / 2 ,16)
    w_conv1 = nn.set_weight(shape=[4, 4, 1, 16])
    b_conv1 = nn.set_bias(shape=[16])
    layer1 = nn.add_conv_layer(x, w_conv1, stride=[1, 1, 1, 1], padding='SAME') + b_conv1
    layer1_relu = tf.nn.relu(layer1)
    layer1_relu_pool = nn.add_pooling(layer1_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    # (7049, height / 2, width / 2, 16) > (7049, height / 4, width / 4, 32)
    w_conv2 = nn.set_weight(shape=[4, 4, 16, 32])
    b_conv2 = nn.set_bias(shape=[32])
    layer2 = nn.add_conv_layer(layer1_relu_pool, w_conv2, stride=[1, 1, 1, 1], padding='SAME') + b_conv2
    layer2_relu = tf.nn.relu(layer2)
    layer2_relu_pool = nn.add_pooling(layer2_relu, stride=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    # (7049, height / 4, width / 4, 32) > (7049, 64)
    fc1_height = height / 4
    fc1_width = width / 4
    if(((height % 4) == 0) & ((width % 4) == 0)):
        w_fc1 = nn.set_weight(shape=[fc1_height * fc1_width * 32, 64])
    elif(((height % 4) == 0) & ((width % 4) != 0)):
        w_fc1 = nn.set_weight(shape=[fc1_height * (fc1_width + 1) * 32, 64])
    elif(((height % 4) != 0) & ((width % 4) == 0)):
        w_fc1 = nn.set_weight(shape=[(fc1_height + 1) * fc1_width * 32, 64])
    else:
        w_fc1 = nn.set_weight(shape=[(fc1_height + 1) * (fc1_width + 1) * 32, 64])
    b_fc1 = nn.set_bias(shape=[64])
    layer3 = tf.matmul(layer2_relu_pool, w_fc1) + b_fc1
    layer3_relu = tf.nn.relu(layer3)
    layer3_drop = tf.nn.dropout(layer3_relu, keep_prob=0.5)

    # (7049, 64) > (7049, coor_data.shape[1])
    w_fc2 = nn.set_weight(shape=[64, coor_data.shape[1]])
    b_fc2 = nn.set_bias(shape=[coor_data.shape[1]])
    layer4 = tf.matmul(layer3_drop, w_fc2) + b_fc2

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(layer4 - y) / ((height + width) / 2), 1))
    train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    return train_op, layer4, cost