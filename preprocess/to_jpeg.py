import numpy as np
from utilis.util import *

x = create_lfw_npy()

# img = tf.placeholder(dtype=tf.float32, shape=[112, 112, 3], name='input_image')

def jpeg_pipe(img):
    before_jpeg = (img+1.0)*127.5
    jpeg_encode = tf.image.encode_jpeg(tf.cast(before_jpeg, dtype=tf.uint8), format='rgb', quality=75)
    after_jpeg = tf.cast(tf.image.decode_jpeg(jpeg_encode), dtype=tf.float32)
    jpeg_decode = after_jpeg/127.5 - 1.0
    return jpeg_decode

num_img = len(x)
array = np.zeros([num_img, 112, 112, 3])


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for it, i in enumerate(x):
        img_jpg = sess.run(jpeg_decode, feed_dict={img:i})
        array[it] = img_jpg

np.save('/data/jiaming/datasets/faces/faces_emore/after_jpeg_lfw.npy', array)
