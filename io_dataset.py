import tensorflow as tf
import os
import numpy as np
from PIL import Image
import imageio
import random


import threading
import time


class PatchScoreDataset(object):
	def __init__(self, dataset_path, class_num = 2):
		self.dataset_path = dataset_path
		self.class_num = class_num
		self.read_patches_directory()

	def get_class_num(self):
		return self.class_num
		
	
	def read_patches_directory(self):
		self.strong_positive_train = [line.rstrip('\n') for line in open(self.dataset_path + "strong_positive_train_samples.txt")]				
		self.strong_positive_test = [line.rstrip('\n') for line in open(self.dataset_path + "strong_positive_test_samples.txt")]		
		self.strong_negative_train = [line.rstrip('\n') for line in open(self.dataset_path + "strong_negative_train_samples.txt")]
		self.strong_negative_test = [line.rstrip('\n') for line in open(self.dataset_path + "strong_negative_test_samples.txt")]
		
		self.weak_train = [line.rstrip('\n') for line in open(self.dataset_path + "weak_train_samples.txt")]
		self.weak_test = [line.rstrip('\n') for line in open(self.dataset_path + "weak_test_samples.txt")]
		
	def image_augmentation(self, img, idx_flip_flop = 0, idx_flop = 0, idx_rot = 0):

		if(idx_flip_flop == 1):
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
		elif(idx_flip_flop == 2):
			img = img.transpose(Image.FLIP_TOP_BOTTOM)
			
		if(idx_rot == 1):
			img = img.transpose(Image.ROTATE_90)
		elif(idx_rot == 2):
			img = img.transpose(Image.ROTATE_180)
		elif(idx_rot == 3):
			img = img.transpose(Image.ROTATE_270)
			
		return img


	def next(self, test):
		
		
		if self.class_num == 3:
			rand_idx = random.randint(0, 2)
			if(test == True):
				if rand_idx == 0:
					rand_img_path = random.choice(self.strong_positive_test)
					score = [1, 0, 0]
				elif rand_idx == 1:
					rand_img_path = random.choice(self.strong_negative_test)
					score = [0, 1, 0]
				else:
					rand_img_path = random.choice(self.weak_test)
					score = [0, 0, 1]
			else:
				if rand_idx == 0:
					rand_img_path = random.choice(self.strong_positive_train)
					score = [1, 0, 0]
				elif rand_idx == 1:
					rand_img_path = random.choice(self.strong_negative_train)
					score = [0, 1, 0]
				else:
					rand_img_path = random.choice(self.weak_train)
					score = [0, 0, 1]
		elif self.class_num == 2:			
			if(test == True):
				if random.randint(0, 1) == 0:
					rand_img_path = random.choice(self.strong_positive_test)
					score = [1, 0]
				else:
					#if random.randint(0, 1) == 0:
					rand_img_path = random.choice(self.strong_negative_test)
					#else:
					#	rand_img_path = random.choice(self.weak_test)
					score = [0, 1]
			else:
				if random.randint(0, 1) == 0:
					rand_img_path = random.choice(self.strong_positive_train)
					score = [1, 0]
				else:
					#if random.randint(0, 1) == 0:
					rand_img_path = random.choice(self.strong_negative_train)
					#else:
					#	rand_img_path = random.choice(self.weak_train)
					score = [0, 1]
		
			
		raw_patch = Image.open(os.path.join(self.dataset_path, rand_img_path))
		
		
		img1 = raw_patch.crop((0, 0, 48, 48))
		img2 = raw_patch.crop((48, 0, 96, 48))

		
		aug_flip_flop = np.random.randint(3)
		aug_rotation = np.random.randint(4)

		img1 = self.image_augmentation(img1, aug_flip_flop, aug_rotation)        
		img2 = self.image_augmentation(img2, aug_flip_flop, aug_rotation)
		
		
		
		img1 = np.array(img1)
		img2 = np.array(img2)
		
		#normalization
		img1 = img1.astype(np.float32)
		img2 = img2.astype(np.float32)
		img1 = ((img1/255.)-0.5)*2.
		img2 = ((img2/255.)-0.5)*2.
		
		img1 = np.reshape(img1, (1, 48, 48, 3))
		img2 = np.reshape(img2, (1, 48, 48, 3))

		score = np.array([score])
				
		imgs = np.concatenate((img1, img2), axis = 3)
		
		return imgs, score


	def next_batch(self, batch_size, test=False):
		imgs, score = self.next(test)
		for i in range(batch_size-1):
			imgs_, score_ = self.next(test)
			imgs = np.concatenate([imgs, imgs_], axis=0)
			score = np.concatenate([score, score_], axis=0)

		return imgs, score








class IO_runner(object):
	def __init__(self, shape, cdb):
		(self.batch_size, _, _, _) = shape
		self.shape_imgs = shape
		self.cdb = cdb
		self.shape_labels = [self.batch_size, self.cdb.get_class_num()]
		
		self.imgs = tf.placeholder(tf.float32, self.shape_imgs)
		self.labels = tf.placeholder(tf.float32, self.shape_labels)

		self.queue = tf.FIFOQueue(capacity = 16, dtypes=[tf.float32, tf.float32], shapes=[self.shape_imgs, self.shape_labels])
		
		self.enqueue_op = self.queue.enqueue([self.imgs, self.labels])
		self.queue_size = 0

	def inc_queue_size(self):
		self.queue_size += 1
		
	def dec_queue_size(self):
		self.queue_size -= 1

	def get_queue_size(self):
		return self.queue_size

		
	def get_inputs(self):		
		return self.queue.dequeue()

	def thread_main(self, sess, thr_idx):
		while True:
			is_training = True
			(imgs, labels) = self.cdb.next_batch(self.batch_size, is_training)
			sess.run(self.enqueue_op, feed_dict={self.imgs:imgs, self.labels:labels})
			self.inc_queue_size()			
			time.sleep(0.01)

	def start_threads(self, sess, n_threads=4):
		threads = []
		for n in range(n_threads):
			t = threading.Thread(target=self.thread_main, args=(sess, n))
			t.daemon = True # thread will close when parent quits
			t.start()
			threads.append(t)
			
		return threads
		
		
