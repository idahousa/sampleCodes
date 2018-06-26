#Use this program along with the "train_save_tf_model.py".
#This function use the result by the above train-save script.
import tensorflow as tf
import os
#Place to save the result
save_path = os.path.join(os.getcwd(),'trained_model\\mynet.ckpt')
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

predict(save_path,[[0.5]])

#predict(save_path,[2.])
