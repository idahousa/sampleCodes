import numpy as np
import tensorflow as tf
from sklearn import datasets

train_ratio = 0.8
batch_size = 16
num_epoches = 100
images, labels = datasets.load_digits(return_X_y=True)
num_image = len(labels)
num_train_image = np.int32(num_image*train_ratio)
train_images = images[1:num_train_image,:]
train_labels = labels[1:num_train_image]
valid_images = images[num_train_image:num_image,:]
valid_labels = labels[num_train_image:num_image]
print(train_images.shape)
print(train_labels.shape)
print(valid_images.shape)
print(valid_labels.shape)

dx_train = tf.data.Dataset.from_tensor_slices(train_images)
dy_train = tf.data.Dataset.from_tensor_slices(train_labels).map(lambda x:tf.one_hot(x,10))
train_dataset = tf.data.Dataset.zip((dx_train,dy_train)).shuffle(500).repeat(count=num_epoches).batch(batch_size=batch_size)

dx_valid = tf.data.Dataset.from_tensor_slices(valid_images)
dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda x:tf.one_hot(x,10))
valid_dataset = tf.data.Dataset.zip((dx_valid,dy_valid)).batch(batch_size=1)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
next_element =iterator.get_next()
train_init_op = iterator.make_initializer(train_dataset)
valid_init_op = iterator.make_initializer(valid_dataset)

def nn_network(in_data):
    bn = tf.layers.batch_normalization(inputs=in_data)
    fc1 = tf.layers.dense(inputs=bn,units=256,activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1,units=256,activation=tf.nn.relu)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(inputs=fc2,units=10)
    return fc3

logits = nn_network(next_element[0])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1],logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
prediction = tf.argmax(logits,1)
equality = tf.equal(prediction,tf.argmax(next_element[1],1))
accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))
init_op  = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    session.run(train_init_op)
    iteration = 0
    while True:
        try:
            _,_acc = session.run([optimizer, accuracy])
            iteration +=1
            if iteration%50==0:
                print('Iteration: {}, train acc: {}'.format(iteration,_acc))
        except tf.errors.OutOfRangeError:
            print('Finish training!')
            break
    session.run(valid_init_op)
    valid_acc = 0
    iteration = 0
    while True:
        try:
            _valid_acc = session.run([accuracy])  # type: object
            iteration += 1
            valid_acc += _valid_acc[0]
            print(iteration)
        except tf.errors.OutOfRangeError:
            print('Validation error is: {}'.format(valid_acc*100/iteration))
            break
print('---------------------------------------------------------------------------------------------------------------')
