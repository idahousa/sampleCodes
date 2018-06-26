import tensorflow as tf
import numpy as np
import os
#Place to save the result
save_path = os.path.join(os.getcwd(),'trained_model\\')
if not os.path.exists(save_path):
    os.mkdir(save_path)
print(save_path)
#Create fake data for regression
npx = np.random.uniform(low=-1.0,high=1.0,size=[1000,1]) #Create a 1000-by-1 matrix as the x value
noise = np.random.normal(loc=0.0,scale=0.001,size=npx.shape)
npy = npx*2 + noise
#Separating the train and test sets
npx_train, npx_test = np.split(npx,[800])
npy_train, npy_test = np.split(npy,[800])

tfx = tf.placeholder(dtype=tf.float32,shape=npx_train.shape)
tfy = tf.placeholder(dtype=tf.float32,shape=npy_train.shape)
#Create dataset object
dataset = tf.data.Dataset.from_tensor_slices((tfx,tfy))
dataset = dataset.shuffle(buffer_size=800)
dataset = dataset.batch(batch_size=32)
dataset = dataset.repeat(count=1000)
dataset = dataset.map(lambda x,y: (x,y*0.5))
iterator = dataset.make_initializable_iterator()
print('-'*100)
print(dataset.output_classes)
print(dataset.output_shapes)
print(dataset.output_types)
print('-'*100)
#Define the network
(bx, by) = iterator.get_next()
l1 = tf.layers.dense(inputs=bx,units=1024,activation=tf.nn.relu,name='layer_1')
l2 = tf.layers.dense(inputs=l1, units=1024, activation=tf.nn.relu,name='layer_2')
output = tf.layers.dense(inputs=l2,units=1,name='output_layer')

loss = tf.losses.mean_squared_error(labels=by,predictions=output)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
session = tf.Session()
session.run(fetches=[iterator.initializer, tf.global_variables_initializer()],feed_dict={tfx:npx_train,tfy:npy_train})
saver = tf.train.Saver()
for step in range(5000):
    try:
        _,train_loss = session.run(fetches=[train, loss])
        if step%100 ==0:
            test_loss = session.run(fetches=[loss],feed_dict={bx:npx_test, by:npy_test})
            print('Train Step: {} - Train loss:{}- Test Loss: {}'.format(step,train_loss, test_loss))
    except tf.errors.OutOfRangeError: #This error is raised when iterator reaches out of dataset
        print('Finish training procedure')
        break

print('Saving the trained model TO...')
saved_path = saver.save(sess=session, save_path=save_path + 'mynet.ckpt',write_meta_graph=False)
print(saved_path)
print('Saved the trained model!')
session.close()
tf.reset_default_graph()
#Restore the model
def predict(model_path, x_data):
    tfx = tf.placeholder(dtype=tf.float32, shape=[1,1])
    l1 = tf.layers.dense(inputs=tfx, units=1024, activation=tf.nn.relu, name='layer_1')
    l2 = tf.layers.dense(inputs=l1, units=1024, activation=tf.nn.relu, name='layer_2')
    output = tf.layers.dense(inputs=l2, units=1, name='output_layer')
    session = tf.Session()
    restorer = tf.train.Saver()
    restorer.restore(session,save_path=model_path)
    pred = session.run(fetches=[output],feed_dict={tfx:x_data})
    print(pred)
    return pred

print(predict(saved_path,[[0.8]]))
