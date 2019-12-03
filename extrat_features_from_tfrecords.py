#coding=utf-8
import tensorflow as tf
import numpy as np

tfrecords_path = '/media/civic/sdc/cyx/Attention/tfrecords/'

tf_all = tfrecords_path + 'ucf11_train_split2.tfrecords'
tf1 = tfrecords_path+'ucf11_train_split_11.tfrecords'
tf2 = tfrecords_path+'ucf11_train_split_22.tfrecords'
tf3 = tfrecords_path+'ucf11_train_split_33.tfrecords'
#tf4 = tfrecords_path+'train_split4.tfrecords'
tfrecords_list = [tf1, tf2, tf3]
#batch_size = 32
def read_single_sample(filename):
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer(filename, num_epochs=None)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # get feature from serialized example

    features = tf.parse_single_example(
        serialized_example,
        features={
            'feature_batch': tf.FixedLenFeature([], tf.string),
            'label_batch': tf.FixedLenFeature([], tf.int64)
        }
    )

    label_batch = features['label_batch']

    fbm_raw = features['feature_batch']
    fbm_raw = tf.decode_raw(fbm_raw, tf.float32)
    feature_batch = tf.reshape(fbm_raw, [30,8,8,2048])

    return feature_batch,label_batch



def generate_feature_batch_and_label_batch(batch_size):

    example_list = [read_single_sample(tfrecords_list) for _ in range(6)]
    example_batch, label_batch =tf.train.shuffle_batch_join(example_list,batch_size=batch_size,capacity=512, min_after_dequeue=256)
    label_batch = tf.reshape(label_batch,[batch_size, 1])
    #   print label_batch.get_shape( )
    # y = tf.cast(label_batch,dtype=tf.float32)
    # x = tf.ones([1, 30],tf.float32)
    # out=tf.matmul(y, x,
    #        transpose_a=False, transpose_b=False,
    #        a_is_sparse=False, b_is_sparse=False)
    #print out.get_shape()
    #out = tf.cast(out,dtype=tf.int64)
    print "batch label size", label_batch.get_shape()
    out = tf.tile(label_batch,[1,30])
    print "batch data size", out.get_shape()
    return example_batch, out

