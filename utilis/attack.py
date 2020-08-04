import tensorflow as tf
import numpy as np
import yaml

IMAGE_SHAPE = 112
prob = 0.5
epsilon = 8 / 255. * 2
step_size = 2 / 255. * 2
niter = 10
bounds = (-1, 1)

config = yaml.load(open('./configs/config_ms1m_100.yaml'),Loader=yaml.FullLoader)

def FGSM(x, dist, eps=epsilon):
    x_adv = x + eps * tf.sign(tf.gradients(dist, x)[0])
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)

def FGSM2(x, model_function, dist_function, perturbation_multiplier=1):
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < 5

    def loop_body(index, adv_images):

        adv_images_di = input_diversity(adv_images)
        tmp_embd = model_function(adv_images_di)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result


def IFGSM(x, model_function, dist_function, perturbation_multiplier=1):
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):

        tmp_embd,_ = model_function(adv_images)
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def I2FGSM(x, model_function, dist_function, perturbation_multiplier=1):
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    loop_vars = [0, start_x]

    def loop_cond(index, _):
        return index < niter

    def loop_body(index, adv_images):
        adv_images_di = input_diversity(adv_images)
        tmp_embd = model_function(tf.reshape(adv_images_di, [-1, 112 ,112 ,3]))
        dist = dist_function(tmp_embd)
        grad = tf.gradients(dist, adv_images)[0]
        perturbation = step_size * tf.sign(grad)
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    _, result = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars,
        back_prop=False,
        parallel_iterations=1)
    return result

def MIFGSM(x, model_function, dist_function, momentum = 1, perturbation_multiplier=1):
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]


    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):
        tmp_embd, _ = model_function(adv_images)
        dist = dist_function(tmp_embd)
        noise = tf.gradients(dist, adv_images)[0]
        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

def MI2FGSM(x, model_function, dist_function, momentum = 1, perturbation_multiplier=1):
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    start_x = x #+ tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    start_grad = tf.zeros(tf.shape(x))
    loop_vars = [0, start_x, start_grad]


    def loop_cond(index, _, __):
        return index < niter

    def loop_body(index, adv_images, grad):

        adv_images_di = input_diversity(adv_images)
        tmp_embd, _ = model_function(adv_images_di)
        dist = dist_function(tmp_embd)
        noise = tf.gradients(dist, adv_images)[0]
        noise = noise / tf.reduce_mean(tf.abs(noise), axis=0, keep_dims=True)
        noise = momentum * grad + noise
        perturbation = step_size * tf.sign(noise)

        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images, noise

    with tf.control_dependencies([start_x]):
        _, result, _ = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
    return result

## image_resize值应该是下一个网络输入的tenor的shape[0]
def input_scaled(input_tensor):
    rnd = tf.random_uniform((), int(IMAGE_SHAPE-2), IMAGE_SHAPE, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = IMAGE_SHAPE - rnd
    w_rem = IMAGE_SHAPE - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], IMAGE_SHAPE, IMAGE_SHAPE, 3))

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: padded, lambda: input_tensor)
#

def input_rotate(input_tensor):
    random_angles = tf.random.uniform(shape=(tf.shape(input_tensor)[0],), minval=-np.pi / 12, maxval=np.pi / 12)

    rotated_images = tf.contrib.image.transform(
        input_tensor,
        tf.contrib.image.angles_to_projective_transforms(
            random_angles, tf.cast(tf.shape(input_tensor)[1], tf.float32), tf.cast(tf.shape(input_tensor)[2], tf.float32)
        ))

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda: rotated_images, lambda: input_tensor)
#

def input_enhance(input_tensor):
    max_pixel = 5 / 127.5 - 1.0
    min_pixel = -5 / 127.5 - 1.0
    random_pixel = tf.random.uniform(shape=tf.shape(input_tensor),minval = min_pixel , maxval = max_pixel)
    random_pixel = random_pixel/127.5-1
    pixel_image = input_tensor + random_pixel
    pixel_image = tf.clip_by_value(pixel_image, -1, 1)

    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(prob), lambda:pixel_image , lambda: input_tensor,tf.float32)
#

def input_diversity(input_tensor):


    input_tensor = input_enhance(input_tensor)
    input_tensor = input_scaled(input_tensor)
    input_tensor = input_rotate(input_tensor)



    return tf.reshape(input_tensor, [-1, 112 ,112 ,3])
