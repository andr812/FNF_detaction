import os


import tensorflow as tf

from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import numpy as np
import argparse

import io_dataset


import time



image_size = 48
img_channels = 3


# Hyperparameter
init_learning_rate = 1e-4

weight_decay = 1e-4

# Label & batch_size
batch_size = 64

iteration = 200
# batch_size * iteration = data_set_number

test_iteration = 20
total_epochs = 300000000000


parser = argparse.ArgumentParser(description='Training Detector Network')
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu_id)
		
		


print("******************************")
print('class_num=', args.class_num)
print("******************************")

DATASET_PATH = "/home/thkim/data/DIV2K/"
dataset = io_dataset.PatchScoreDataset(DATASET_PATH, args.class_num)






DEPTH = 40
N = int((DEPTH - 4)  / 3)
GROWTHRATE =12
		
def conv(name, l, channel, stride, kernel_size=3):	
		with tf.name_scope(name):
			return tf.layers.conv2d(inputs=l, use_bias=False, filters=channel, kernel_size=kernel_size, strides=stride, padding='SAME')


def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=args.class_num, name='linear')


def add_layer(name, l, is_training):
		shape = l.get_shape().as_list()
		in_channel = shape[3]
		with tf.variable_scope(name) as scope:
			#c = BatchNorm('bn1', l)
			c = Batch_Normalization(l, training=is_training, scope=name+'bn1')
			c = tf.nn.relu(c)
			c = conv('conv1', c, GROWTHRATE, 1)
			l = tf.concat([c, l], 3)
		return l

def add_transition(name, l, is_training):
	shape = l.get_shape().as_list()
	in_channel = shape[3]
	with tf.variable_scope(name) as scope:
		#l = BatchNorm('bn1', l)
		l = Batch_Normalization(l, training=is_training, scope=name+'bn1')
		l = tf.nn.relu(l)
		#l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
		l = conv('conv1', l, in_channel, stride=1, kernel_size = 3)
		l = tf.nn.relu(l)
		#l = AvgPooling('pool', l, 2)
		l = tf.layers.average_pooling2d(l, (2, 2), (2, 2))


	return l
		
		
		

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        #test_batch_x = test_x[test_pre_index: test_pre_index + add]
        #test_batch_y = test_y[test_pre_index: test_pre_index + add]
        #test_pre_index = test_pre_index + add
        (test_imgs, test_batch_y) = dataset.next_batch(batch_size, test=True)

        test_feed_dict = {
            x: test_imgs,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration
    test_acc /= test_iteration

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary



		
		
class DenseNet():
	def __init__(self, x, training):
		self.training = training
		self.model = self.dense_net(x)




	
			
	


	def dense_net(self, image):
		l = conv('conv0', image, 16, 1)
		with tf.variable_scope('block1') as scope:

			for i in range(N):
				l = add_layer('dense_layer.{}'.format(i), l, self.training)
			l = add_transition('transition1', l, self.training)

		with tf.variable_scope('block2') as scope:

			for i in range(N):
				l = add_layer('dense_layer.{}'.format(i), l, self.training)
			l = add_transition('transition2', l, self.training)

		with tf.variable_scope('block3') as scope:

			for i in range(N):
				l = add_layer('dense_layer.{}'.format(i), l, self.training)


		
		x = Batch_Normalization(l, training=self.training, scope='linear_batch')
		x = Relu(x)
		x = Global_Average_Pooling(x)
		x = flatten(x)
		logits = Linear(x)

		

		return logits
			
			
			
			
			

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels*2])
label = tf.placeholder(tf.float32, shape=[None, args.class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=x, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
In paper, use MomentumOptimizer
init_learning_rate = 0.1
but, I'll use AdamOptimizer
"""

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost + l2_loss * weight_decay)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




save_point = 10
saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
save_path = './model_class=' + str(args.class_num) + '_gpu_id=' + str(args.gpu_id) + '/'

print(os.path.isdir(save_path))
if(os.path.isdir(save_path) == False):
	os.mkdir(save_path)
print("*************************")
print(save_path)







with tf.Session() as sess:

	ckpt = tf.train.get_checkpoint_state(save_path)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("restored")
	else:
		sess.run(tf.global_variables_initializer())
		print("inited")

	summary_writer = tf.summary.FileWriter('./logs'+ str(args.class_num), sess.graph)

	epoch_learning_rate = init_learning_rate
	for epoch in range(1, total_epochs + 1):
		#if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
		#    epoch_learning_rate = epoch_learning_rate / 10

		pre_index = 0
		train_acc = 0.0
		train_loss = 0.0


		for step in range(1, iteration + 1):
			'''
			if pre_index+batch_size < 50000 :
				batch_x = train_x[pre_index : pre_index+batch_size]
				batch_y = train_y[pre_index : pre_index+batch_size]
			else :
				batch_x = train_x[pre_index : ]
				batch_y = train_y[pre_index : ]

			batch_x = data_augmentation(batch_x)
			'''
			imgs, batch_y = dataset.next_batch(batch_size)
			train_feed_dict = {
				x: imgs,
				label: batch_y,
				learning_rate: epoch_learning_rate,
				training_flag : True
			}

			_, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
			batch_acc = accuracy.eval(feed_dict=train_feed_dict)

			train_loss += batch_loss
			train_acc += batch_acc
			pre_index += batch_size

			if step == iteration :
				train_loss /= iteration # average loss
				train_acc /= iteration # average accuracy

				train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
												  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

				test_acc, test_loss, test_summary = Evaluate(sess)

				summary_writer.add_summary(summary=train_summary, global_step=epoch)
				summary_writer.add_summary(summary=test_summary, global_step=epoch)
				summary_writer.flush()

				line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
					epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
	
				print(line)

				with open(save_path + 'logs' + str(args.class_num) + '.txt', 'a') as f :
					f.write(line)
			
		if epoch % 10 == 0:
			saver.save(sess=sess, save_path= save_path + 'dense.ckpt')

