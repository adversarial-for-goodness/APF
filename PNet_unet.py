from backbones.insightface import *
from utilis.util import *
from utilis.attack import *
from backbones.unet import unet
from backbones.MobileFaceNet import mobilefacenet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def mobile(inputs):
    prelogits, net_points = mobilefacenet(inputs, bottleneck_layer_size=192, reuse=tf.AUTO_REUSE)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    return embeddings


def train(args):
    # args = get_args()
    epoch = args.train_epoch
    batch_size = args.train_batchsize
    config = yaml.load(open(args.config_path))
    benchmark = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_benchmark')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
    benchmark_embds = mobile(benchmark)
    embds = mobile(images)

    def get_distance(embds1, embds2=benchmark_embds):
        embeddings1 = embds1 / tf.norm(embds1, axis=1, keepdims=True)
        embeddings2 = embds2 / tf.norm(embds2, axis=1, keepdims=True)
        diff = tf.subtract(embeddings1, embeddings2)
        distance = tf.reduce_sum(tf.multiply(diff, diff), axis=1)
        return distance

    arc_embds, _ = get_embd(images, config)
    arc_ben_embds, _ = get_embd(benchmark, config)



    # adversarial
    # grad_op = tf.gradients(dist, inputs)[0]
    # x_fgsm = FGSM(inputs_placeholder, dist)

    # x_ifgsm = IFGSM(inputs_placeholder, lambda f_embd: get_embd(f_embd),
    #                                 lambda f_dis: get_distance(f_dis), 1)
    # x_mifgsm = MI2FGSM(inputs_placeholder, lambda f_embd: get_embd(f_embd),
    #                                 lambda f_dis: get_distance(f_dis), 1)
    x_i2fgsm = I2FGSM(images, lambda f_embd: mobile(f_embd), lambda f_dis: get_distance(f_dis), 1)
    # x_mi2fgsm = MI2FGSM(inputs_placeholder, lambda f_embd: get_embd(f_embd), lambda f_dis: get_distance(f_dis),1)

    x_adv = x_i2fgsm
    x_noise = x_adv - images


    distances = get_distance(arc_embds, arc_ben_embds)
    threshold = 1.02
    distances = threshold - distances
    prediction = tf.sign(distances)
    correct_prediction = tf.count_nonzero(prediction+1, dtype=tf.float32)
    accuracy = correct_prediction/batch_size

    output = unet(x_noise)
    loss_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='APF')

    eps = 8 / 255. * 2
    s = tf.clip_by_value(output, - eps, eps)
    image_adv = tf.clip_by_value(images + s, -1.0, 1.0)
    embds_adv, _ = get_embd(image_adv, config)
    embds_ben_arc, _ = get_embd(benchmark, config)
    distances_adv_s = get_distance(embds_adv, embds_ben_arc)
    loss_distance_s = 10 - distances_adv_s
    loss_s = tf.reduce_mean(loss_distance_s)

    loss = loss_s
    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss, var_list=loss_vars)
    # accuracy = accurate(x, y)

    variables_unet = tf.contrib.framework.get_variables_to_restore(include=['APF'])
    saver_unet = tf.train.Saver(variables_unet)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        variables_mobilefacenet = tf.contrib.framework.get_variables_to_restore(include=['MobileFaceNet'])
        saver_m = tf.train.Saver(variables_mobilefacenet)
        variables_arc = tf.contrib.framework.get_variables_to_restore(include=['embd_extractor'])
        saver_a = tf.train.Saver(variables_arc)

        # the path you save the Mobilefacenet model.
        saver_m.restore(sess, args.mobilefacenet_model_path)
        saver_a.restore(sess, args.insightface_model_path)

        X = create_lfw_npy()
        print(X.shape)
        X_0 = X[0::2]
        X_1 = X[1::2]
        num_test = len(X_0)
        # the path you save the train data.
        train_dataset = np.load(args.train_data,allow_pickle=True)

        list_img = np.array(train_dataset) / 127.5 - 1.0
        list_img = np.reshape(list_img, [-1, 112, 112, 3])
        list_img_0 = list_img[0::2]
        list_img_1 = list_img[1::2]
        len_train = len(list_img_0)


        n_batch = len_train//batch_size

        # saver = tf.train.Saver(max_to_keep=1)
        best_acc = 1.0
        for i in range(epoch):
            for batch in range(n_batch):
                x_batch = list_img_0[batch*batch_size:(batch+1)*batch_size]
                y_batch = list_img_1[batch * batch_size:(batch + 1) * batch_size]

                sess.run(train_op, feed_dict={
                                           images: x_batch,
                                           benchmark: y_batch})

                if batch % 50 == 0:
                    train_loss = sess.run(loss, feed_dict={images:x_batch, benchmark:y_batch})
                    acc = 0
                    test_loss = 0
                    for j in range(num_test//batch_size):

                        x = X_0[j*batch_size:(j+1)*batch_size]
                        ben = X_1[j * batch_size:(j + 1) * batch_size]
                        x_adv, test_loss_temp = sess.run([image_adv, loss], feed_dict={images:x, benchmark:ben})
                        test_loss = test_loss + test_loss_temp
                        acc += sess.run(accuracy, feed_dict={images: x_adv, benchmark: ben})
                    print('Epoch {}, iter {}, train_loss={:.4}, test_loss={:.4}, acc={:.4}'.format(i, batch, train_loss, test_loss/(num_test//batch_size), acc/(num_test//batch_size)))
                    temp = acc / (num_test // batch_size)
                    if temp < best_acc:
                        best_acc = temp
                        saver_unet.save(sess,args.train_model_ouput)
                        # saver_unet.save(sess,'/data/jiaming/code/InsightFace-tensorflow/model/mm/test_zx/model_apf' + str(best_acc) +'.ckpt')
                        # print(best_acc)

