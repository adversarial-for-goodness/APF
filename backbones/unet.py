import tensorflow as tf
import numpy as np

universal = np.load('/data/jiaming/code/InsightFace-tensorflow/data/precomputing_perturbations/perturbation_wuqiongfanshu.npy')

channel_list = [64, 128, 256, 512]
channel_num = len(channel_list)
def unet(x_input, scope=None, reuse=None, universal=universal):
    with tf.variable_scope('APF', reuse=reuse):
        x_down = [x_input]
        for i, item in enumerate(channel_list):
            x = tf.layers.conv2d(
                x_down[i],
                filters=item,
                strides=2,
                kernel_size=[3,3],
                padding='SAME',
                activation=tf.nn.leaky_relu
            )
            # x = tf.layers.batch_normalization(x, training=train_placeholder)
            x_down.append(x)

        channel_list_reverse = [256, 128, 64]
        x_up = [x_down[-1]]
        for i, item in enumerate(channel_list_reverse):
            # x_shape = x.get_shape().as_list()

            x = tf.layers.conv2d_transpose(
                x_up[i],
                filters=item,
                strides=2,
                kernel_size=[3,3],
                padding='SAME',
                activation=tf.nn.leaky_relu
            )
            # x = tf.layers.batch_normalization(x, training=train_placeholder)
            x_up.append(tf.concat([x, x_down[-(i+2)]],axis=-1))


        output = tf.layers.conv2d_transpose(
                x_up[-1],
                filters=3,
                strides=2,
                kernel_size=[3,3],
                padding='SAME',
                activation=None
            )
        # E Module

        universal = tf.convert_to_tensor(universal, tf.float32)
        scale_alpha = tf.Variable(tf.constant(1.0))
        scale_beta = tf.Variable(tf.constant(0.0))
        universal = universal * scale_alpha + scale_beta

        noise = tf.layers.conv2d(inputs=universal,
                                 filters=3,
                                 kernel_size=[1, 1],
                                 strides=1,
                                 padding='valid',
                                 activation=None,
                                 use_bias=True
                                 )

        output = noise + output

        return output
