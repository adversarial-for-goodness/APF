import tensorflow as tf
import tensorflow.contrib.slim as slim
from backbones import modifiedResNet_v2, ResNet_v2
import math
import yaml
import pickle
import argparse
from tensorflow.python.framework import graph_util

W_INIT = tf.contrib.layers.xavier_initializer(uniform=False)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='./model/ms1m/best-m-334000', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')

    return parser.parse_args()


def get_embd(inputs, config, reuse=tf.AUTO_REUSE, scope='embd_extractor'):
    is_training_dropout = False
    is_training_bn = False
    with tf.variable_scope(scope, reuse=reuse):
        net = inputs
        end_points = {}
        if config['backbone_type'].startswith('resnet_v2_m'):
            arg_sc = modifiedResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_m_50':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_101':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_152':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_m_200':
                    net, end_points = modifiedResNet_v2.resnet_v2_m_200(net, is_training=is_training_bn, return_raw=True)
                else:
                    raise ValueError('Invalid backbone type.')
        elif config['backbone_type'].startswith('resnet_v2'):
            arg_sc = ResNet_v2.resnet_arg_scope(weight_decay=config['weight_decay'], batch_norm_decay=config['bn_decay'])
            with slim.arg_scope(arg_sc):
                if config['backbone_type'] == 'resnet_v2_50':
                    net, end_points = ResNet_v2.resnet_v2_50(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_101':
                    net, end_points = ResNet_v2.resnet_v2_101(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_152':
                    net, end_points = ResNet_v2.resnet_v2_152(net, is_training=is_training_bn, return_raw=True)
                elif config['backbone_type'] == 'resnet_v2_200':
                    net, end_points = ResNet_v2.resnet_v2_200(net, is_training=is_training_bn, return_raw=True)
        else:
            raise ValueError('Invalid backbone type.')

        if config['out_type'] == 'E':
            with slim.arg_scope(arg_sc):
                net = slim.batch_norm(net, activation_fn=None, is_training=is_training_bn)
                net = slim.dropout(net, keep_prob=config['keep_prob'], is_training=is_training_dropout)
                net = slim.flatten(net)
                net = slim.fully_connected(net, config['embd_size'], normalizer_fn=None, activation_fn=None)
                net = slim.batch_norm(net, scale=False, activation_fn=None, is_training=is_training_bn)
                end_points['embds'] = net
        else:
            raise ValueError('Invalid out type.')

        return net, end_points


def calculate_arcface_logits(embds, weights, labels, class_num, s, m):
    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t = tf.matmul(embds, weights, name='cos_t')

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')
    return output


def get_logits(embds, labels, config, w_init=W_INIT, reuse=False, scope='logits'):
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], config['class_num']], dtype=tf.float32, initializer=w_init, regularizer=slim.l2_regularizer(config['weight_decay']), trainable=True)
        if config['loss_type'] == 'arcface':
            return calculate_arcface_logits(embds, weights, labels, config['class_num'], config['logits_scale'], config['logits_margin'])
        elif config['loss_type'] == 'softmax':
            return slim.fully_connected(embds, num_outputs=config['class_num'], activation_fn=None, normalizer_fn=None, weights_initializer=w_init, weights_regularizer=slim.l2_regularizer(config['weight_decay']))
        else:
            raise ValueError('Invalid loss type.')

# def input_diversity(input_tensor):
#     rnd = tf.random_uniform((), 112, 130, dtype=tf.int32)
#     rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     h_rem = 130 - rnd
#     w_rem = 130 - rnd
#     pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
#     pad_bottom = h_rem - pad_top
#     pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
#     pad_right = w_rem - pad_left
#     padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
#                     constant_values=0.)
#     padded.set_shape((input_tensor.shape[0], 130, 130, 3))
#     return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.5), lambda: padded,
#                    lambda: input_tensor)



if __name__ == '__main__':
    args = get_args()
    config = yaml.load(open(args.config_path))
    images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
    # img = input_diversity(images)
    embds, _ = get_embd(images, config)
    # b = tf.Variable(1.0, name='b',dtype=tf.float32)
    op = tf.add(embds, 0.0, name='op_to_store_a')
    print('done!')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        print('loading...')
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['op_to_store_a:0'])
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver(var_list=[v for v in tf.global_variables()])
        saver.restore(sess, args.model_path)
        print('restore done!')
        # saver.save(sess, './model/debug/debug_model.ckpt')
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store_a'])

        with tf.gfile.FastGFile('./model/pb/arcface_transform.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
            print('save done!')












# embds, logits, end_points = inference(train_images, train_labels, train_phase_dropout, train_phase_bn, config)


