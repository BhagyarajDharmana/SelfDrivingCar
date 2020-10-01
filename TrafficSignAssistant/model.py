import tensorflow as tf
from tensorflow_core.python.layers.core import flatten
from sklearn.utils import shuffle


def l_net(x):
    """"
    Architecture of the L-Net architecture for training Neural Networks
    :param x: Input for the neural network
    :return model
    """
    mu = 0
    sigma = 0.1

    # Layer 1: Convolution. Input 32x32x1  Output = 28x28x6
    conv1_weights = tf.Variable(tf.random.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_bias = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='VALID') + conv1_bias
    conv1 = tf.nn.relu(conv1)  # Activation
    # Pooling: Input 28x28x6 Output 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer2 : convolution. Input: 14x14x6. Output: 10x10x16

    conv2_weights = tf.Variable(tf.random.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_bias = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID') + conv2_bias
    conv2 = tf.nn.relu(conv2)  # Activation
    # Pooling: Input 10x10x16 Output 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = flatten(conv2)  # Flatten layer
    conv2 = tf.nn.dropout(conv2, rate=0.8)  # Dropout

    # Layer 3: Fully connected layers Input :400 Output: 120
    f_connected1_weights = tf.Variable(tf.random.truncated_normal(shape=[400, 120], mean=mu, stddev=sigma))
    f_connected1_bias = tf.Variable(tf.zeros(120))
    f_connected1 = tf.add(tf.matmul(conv2, f_connected1_weights), f_connected1_bias)

    f_connected1 = tf.nn.relu(f_connected1)  # Activation
    f_connected1 = tf.nn.dropout(f_connected1, rate=0.75)  # Dropout

    # Layer 4: Fully connected Layers Input :120  Output 84
    f_connected2_weights = tf.Variable(tf.random.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    f_connected2_bias = tf.Variable(tf.zeros(84))
    f_connected2 = tf.add(tf.matmul(f_connected1, f_connected2_weights), f_connected2_bias)

    f_connected2 = tf.nn.relu(f_connected2)  # Activation

    # Layer 5 : Fully connected. Input: 84 Output : 43
    f_connected3_weights = tf.Variable(tf.random.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    f_connected3_bias = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(f_connected2, f_connected3_weights), f_connected3_bias)

    return logits


def evaluate(x, y, x_data, y_data, batch_size, loss_operation, accuracy_operation):
    num_examples = len(x_data)
    total_acc = 0
    total_loss = 0
    sess = tf.compat.v1.get_default_session()
    for i in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[i:i + batch_size], y_data[i:i + batch_size]
        loss, accuracy = sess.run([loss_operation, accuracy_operation],
                                  feed_dict={x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        total_acc += (accuracy * len(batch_x))
    return total_loss / num_examples, total_acc / num_examples


def train(x_train, y_train, x_valid, y_valid, learning_rate=0.001, epochs=60, batch_size=32):
    """
    Training the model using L_net architecture
    :param x_train: Training  images
    :param y_train: Training labels
    :param x_valid: Validation images
    :param y_valid: validation labels
    :param learning_rate: Learning rate for optimizer
    :param batch_size: size of the batch
    :param epochs: No.of epochs the model should train
    """
    train_loss_history = []
    valid_loss_history = []
    train_accuracy_history = []
    valid_accuracy_history = []
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.compat.v1.placeholder(tf.int32, shape=None)
    one_hot_y = tf.one_hot(y, 43)
    logits = l_net(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        train_examples = len(x_train)
        print("Training the model........")

        for i in range(epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(0, train_examples, batch_size):
                x_batch, y_batch = x_train[j:j + batch_size], y_train[j:j + batch_size]
                sess.run(training_operation, feed_dict={x: x_batch, y: y_batch})

            validation_loss, validation_accuracy = evaluate(x, y, x_valid, y_valid, batch_size, loss_operation,
                                                            accuracy_operation)

            valid_loss_history.append(validation_loss)
            valid_accuracy_history.append(validation_accuracy)
            train_loss, train_accuracy = evaluate(x, y, x_train, y_train, batch_size, loss_operation,
                                                  accuracy_operation)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)

            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Training Accuracy = {:.3f}".format(train_accuracy))
            print("Validation loss = {:.3f}".format(validation_loss))
            print("Training loss = {:.3f}".format(train_loss))


