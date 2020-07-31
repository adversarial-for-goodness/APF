from utilis.prepare_data import *
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from backbones.MobileFaceNet import mobilefacenet

def normalization(dataset):
    mean = np.mean(dataset)
    std = np.std(dataset)
    dataset = (dataset - mean)
    return mean, std

def jpeg_pipe(img, quality=75):
    before_jpeg = (img+1.0)*127.5
    jpeg_encode = tf.image.encode_jpeg(tf.cast(before_jpeg, dtype=tf.uint8), format='rgb', quality=quality)
    after_jpeg = tf.cast(tf.image.decode_jpeg(jpeg_encode), dtype=tf.float32)
    jpeg_decode = after_jpeg/127.5 - 1.0
    return jpeg_decode

def accurate(distances, batch_size, threshold):
    distances = threshold - distances
    prediction = tf.sign(distances)
    correct_prediction = tf.count_nonzero(prediction + 1, dtype=tf.float32)
    accuracy = correct_prediction / batch_size
    return accuracy

# def accurate(input, groundtruth):
#     assert input.shape == groundtruth.shape
#     input = tf.cast(tf.sign(input), tf.uint8)
#     groundtruth = tf.cast(tf.sign(groundtruth), tf.uint8)
#     num = tf.cast(tf.size(input), tf.float32)
#     sum = tf.reduce_sum(tf.cast(tf.equal(input,groundtruth), tf.float32))
#     return sum/num

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    # model_exp = os.path.join(model_exp, 'MobileFaceNet_9925_9680.pb')
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def my_fooling_rate_calc(v,dataset,f,batch_size=100):
    data0 = dataset[0::2]
    data1 = dataset[1::2]
    dataset_perturbed = data0 + v
    num_images =  np.shape(data0)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.sign(f(data1[m:M, :, :, :],data0[m:M, :, :, :])).flatten()
        est_labels_pert[m:M] = np.sign(f(data1[m:M, :, :, :],dataset_perturbed[m:M, :, :, :])).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    return fooling_rate

def fooling_rate_calc(v,dataset,f,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.sign(f(dataset[m:M, :, :, :],dataset[m:M, :, :, :])).flatten()
        est_labels_pert[m:M] = np.sign(f(dataset[m:M, :, :, :],dataset_perturbed[m:M, :, :, :])).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    return fooling_rate

# def fooling_rate_calc_one(v,dataset,f,get_f,batch_size=100):
#     trainset = dataset[0::2]
#     testset = dataset[1::2]
#     trainset_perturbed = trainset + v
#     num_images =  np.shape(trainset)[0]
#     est_labels_orig = np.zeros((num_images))
#     est_labels_pert = np.zeros((num_images))
#     testset_f = np.zeros([num_images,512])
#     trainset_f = np.zeros([num_images,512])
#     trainset_perturbed_f = np.zeros([num_images,512])
#
#     num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
#
#     # Compute the estimated labels in batches
#     for ii in range(0, num_batches):
#         m = (ii * batch_size)
#         M = min((ii+1)*batch_size, num_images)
#
#         testset_f[m:M] = get_f(testset[m:M]).reshape([-1,512])
#         trainset_f[m:M] = get_f(trainset[m:M]).reshape([-1, 512])
#         testset_f[m:M] = testset_f[m:M]/np.linalg.norm(testset_f[m:M], axis=1, keepdims=True)
#         trainset_f[m:M] = trainset_f[m:M]/np.linalg.norm(trainset_f[m:M], axis=1, keepdims=True)
#         diff = np.subtract(testset_f[m:M], trainset_f[m:M])
#         dist = np.sum(np.square(diff),1)
#         est_labels_orig[m:M] = np.sign(dist).flatten()
#
#         trainset_perturbed_f[m:M] = get_f(trainset_perturbed[m:M]).reshape([-1, 512])
#         trainset_perturbed_f[m:M] = trainset_perturbed_f[m:M]/np.linalg.norm(trainset_f[m:M], axis=1, keepdims=True)
#         diff = np.subtract(testset_f[m:M], trainset_perturbed_f[m:M])
#         dist = np.sum(np.square(diff),1)
#         est_labels_pert[m:M] = np.sign(dist).flatten()
#
#         # est_labels_orig[m:M] = np.sign(f(testset[m:M, :, :, :])).flatten()
#         # est_labels_pert[m:M] = np.sign(f(dataset_perturbed[m:M, :, :, :])).flatten()
#
#     # Compute the fooling rate
#     fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
#     return fooling_rate

def fooling_rate_calc_one(v,dataset,get_f,batch_size=100):
    feature_num = 512
    trainset = dataset[0::2]
    testset = dataset[1::2]
    trainset_perturbed = trainset + v
    num_images =  np.shape(trainset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))
    testset_f = np.zeros([num_images,feature_num])
    trainset_f = np.zeros([num_images,feature_num])
    trainset_perturbed_f = np.zeros([num_images,feature_num])

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        threshold = 1.02
        testset_f[m:M] = get_f(testset[m:M]).reshape([-1,feature_num])
        trainset_f[m:M] = get_f(trainset[m:M]).reshape([-1, feature_num])
        testset_f[m:M] = testset_f[m:M]/np.linalg.norm(testset_f[m:M], axis=1, keepdims=True)
        trainset_f[m:M] = trainset_f[m:M]/np.linalg.norm(trainset_f[m:M], axis=1, keepdims=True)
        diff1 = np.subtract(testset_f[m:M], trainset_f[m:M])

        # 平方和之后开平方，求出欧式距离
        dist1 = np.sum(np.square(diff1),1)

        #阈值与欧式距离的差
        dist1 = threshold - dist1
        est_labels_orig[m:M] = np.sign(dist1).flatten()

        trainset_perturbed_f[m:M] = get_f(trainset_perturbed[m:M]).reshape([-1, feature_num])
        trainset_perturbed_f[m:M] = trainset_perturbed_f[m:M]/np.linalg.norm(trainset_perturbed_f[m:M], axis=1, keepdims=True)
        diff2 = np.subtract(testset_f[m:M], trainset_perturbed_f[m:M])
        dist2 = np.sum(np.square(diff2),1)
        dist2 = threshold - dist2
        est_labels_pert[m:M] = np.sign(dist2).flatten()


        # est_labels_orig[m:M] = np.sign(f(testset[m:M, :, :, :])).flatten()
        # est_labels_pert[m:M] = np.sign(f(dataset_perturbed[m:M, :, :, :])).flatten()

    # Compute the fooling rate
    # face = np.concatenate([trainset_f, testset_f],axis=0)
    # np.save('./data/face_all.npy',face)
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    print("fooling rate is :",fooling_rate)
    # Compute the original rate
    original_rate = float(np.sum(est_labels_orig == 1.) / float(num_images))
    print("original rate is :", original_rate)
    return fooling_rate

def target_fooling_rate_calc(v,dataset,f,target,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate

    target_fooling_rate = float(np.sum(est_labels_pert == target) / float(num_images))
    return target_fooling_rate

def fooling_rate_calc_all(v,dataset,f,target,batch_size=100):
    dataset_perturbed = dataset + v
    num_images =  np.shape(dataset)[0]
    est_labels_orig = np.zeros((num_images))
    est_labels_pert = np.zeros((num_images))

    num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

    # Compute the estimated labels in batches
    for ii in range(0, num_batches):
        m = (ii * batch_size)
        M = min((ii+1)*batch_size, num_images)
        est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
        est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

    # Compute the fooling rate
    fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
    target_fooling_rate = float(np.sum(est_labels_pert == target) / float(num_images))
    return fooling_rate,target_fooling_rate
