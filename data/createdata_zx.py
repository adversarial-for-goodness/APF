# 读取所有tfrecord文件得到dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from utilis.util import *



# 最终的接口是 create_data（）


# 解析dataset的函数, 直接把bytes转换回image, 对应方法1
# def parse_record(raw_record):
#     # 按什么格式写入的, 就要以同样的格式输出
#     keys_to_features = {
#         'image': tf.FixedLenFeature((), tf.string),
#         'label': tf.FixedLenFeature((), tf.string),
#     }
#     # 按照keys_to_features解析二进制的
#
#
#     parsed = tf.parse_single_example(raw_record, keys_to_features)
#
#     image = tf.image.decode_image(tf.reshape(parsed['image'], shape=[112,112,3]), 1)
#     image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
#     image.set_shape([None, None, 1])
#     label = tf.image.decode_image(tf.reshape(parsed['label'], shape=[]), 1)
#     label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
#     label.set_shape([None, None, 1])
#
#     return image, label

#直接把bytes类型的ndarray解析回来, 用decode_raw(),对应方法2
def parse_record(raw_record):
    keys_to_features = {
        'img': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
        'shape':tf.FixedLenFeature([3],tf.int64)
    }
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    #
    #     img = tf.reshape(img, shape=(112, 112, 3))

    image = tf.decode_raw(parsed['img'],tf.uint8)
    # image = tf.to_float(image)
    image = tf.reshape(image, [112, 112, 3])
    label = tf.cast(parsed['label'], tf.int64)
    # label = tf.decode_raw(parsed['label'], tf.uint8)
    # label = tf.to_int32(label)
    # label = tf.reshape(label, [256, 256, 1])

    return image, label


# 根据图片数量还原图片

def generate_image(sess, image_num):

    dataset = tf.data.TFRecordDataset('/data/jiaming/datasets/faces/faces_emore/ms1m.tfrecord')
    # 对dataset中的每条数据, 应用parse_record函数, 得到解析后的新的dataset
    dataset = dataset.map(parse_record)
    dataset = dataset.batch(image_num)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # sess.run(iterator.initializer)
    images, labels = sess.run(next_element)
    return images,labels


# 图像类别是0、1、2、3。。。。。80000多
# 这个是最终函数，传某一类别值（0～84000+），返回这一类别的所有图像
def create_data(sess, class_num):
    images,labels = generate_image(sess, (class_num+1)*200)
    # images = np.zeros([class_num*batch_num,112,112,3])
    indicates = np.where(labels == class_num)
    imgs = images[indicates]
    return imgs,imgs.shape[0]


def pickle_save(sess,col,path):
    images = []
    cnt = col*2
    imgs,labels = generate_image(sess,cnt*200)
    temp = 0
    for i in range(cnt):
        indicates = np.where(labels == i)
        img = imgs[indicates]
        if img.shape[0] >= 50:
            image = img[:50]
            images.append(image)
            temp += 1
        if(temp>=col):
            break
    print(len(images))
    with open(path,'wb') as f:
        pickle.dump(images,f)
    return images



def pickle_load(path):

    with open(path,'rb') as f:
        images = pickle.load(f)
    return images

# 前10类的图像的shape如下所示：
#(110, 112, 112, 3)
# (19, 112, 112, 3)
# (83, 112, 112, 3)
# (15, 112, 112, 3)
# (75, 112, 112, 3)
# (82, 112, 112, 3)
# (103, 112, 112, 3)
# (88, 112, 112, 3)
# (96, 112, 112, 3)
# (35, 112, 112, 3)
# 可以看出来每一类图片数量不一样且差别很大
# 我本来想固定batch_size，让每一类的图像数目相同，但是这样就会浪费很多图片（因为要取数量的最小值）
# 所以现在这个create_data只能返回某一类的图

if __name__ == '__main__':
    with tf.Session() as sess:
        # for i in range(10):
        #     imgs = create_data(sess,i)
        #     print(imgs.shape)
        path = '/data/jiaming/datasets/faces/faces_emore/100000.pkl'
        pickle_save(sess,2000,path)
        images = pickle_load(path)
        print(len(images))
