import tutorials.image.cifar10
import tensorflow as tf

if __name__ == '__main__':

    # FLAGS = tf.app.flags.FLAGS
    # FLAGS.data_dir = 'cifar10_data'
    FLAGS = tf.app.flags.FLAGS
    FLAGS.data_dir = 'cifar10_download/'
    cifar10.maybe_download_and_extract()