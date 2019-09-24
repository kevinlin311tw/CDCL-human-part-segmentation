"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf
import keras
import keras.backend as K
import numpy as np


def resize_images(*args, **kwargs):
    return tf.image.resize_images(*args, **kwargs)



class DeformableDeConv(keras.layers.Layer):
	def __init__(self, kernel_size, stride, filter_num, *args, **kwargs):
		self.stride = stride
		self.filter_num = filter_num
		self.kernel_size =kernel_size
		super(DeformableDeConv, self).__init__(*args,**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		in_filters = self.filter_num
		out_filters = self.filter_num
		self.kernel = self.add_weight(name='kernel',
									  shape=[self.kernel_size, self.kernel_size, out_filters, in_filters],
									  initializer='uniform',
									  trainable=True)

		super(DeformableDeConv, self).build(input_shape)

	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = K.shape(target)
		return tf.nn.conv2d_transpose(source, 
									self.kernel, 
									output_shape=target_shape, 
									strides=self.stride, 
									padding='SAME', 
									data_format='NHWC')
	def get_config(self):
		config = {'kernel_size': self.kernel_size, 'stride': self.stride, 'filter_num': self.filter_num}
		base_config = super(DeformableDeConv, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

'''
class myDeConv(keras.layers.Layer):
	def __init__(self, input_shape,output_shape):

	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel',shape())

    def call(self, inputs, **kwargs):
		source, target, out_filters = inputs
		target_shape = K.shape(target)
		hight = target[1]
		width = target[2]
		in_filters = target[3]
		# W = tf.Variable(tf.random_normal([hight,width,out_filters,in_filters]))
		kernel = tf.constant(1.0, shape=[hight,width,out_filters,in_filters])
		return tf.layers.conv2d_transpose(source, kernel, output_shape=target_shape, strides=2, padding='SAME', data_format='NHWC')

'''

class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class ScalingLayer(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
