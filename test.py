#coding = utf-8
import time
import os
import numpy as np
import tensorflow as tf
import cv2
import sys
import get_features_by_CNN
import train

video_path = '/media/civic/sdc/cyx/Attention/action_youtube_naudio/'

"""Medium config."""
dataset = 'UCF11'
class_num = 11
num_layers = 2
num_steps = 30
hidden_size = 512
keep_prob = 1.0
test_video_num = 652
log_txt_dir = './logs_txt/'
model_dir = '/media/civic/sdc/cyx/Attention/tfs_saver/'
model_name = 'UCF11_comb_gsparse_size_4_step_4_count_1_lamda_0.0001_iteration_6001.ckpt'
test_sample_txt = '/media/civic/sdc/cyx/Attention/action_youtube_naudio/ucf11_test_split_2.txt'

# no groupsparse
# iteration = my_train.max_epoch
# globle_lamda = my_train.globle_lamda

# groupsparse
iteration = train.max_epoch
globle_lamda = train.globle_lamda

g_size = train.group_sparse_size
g_step = train.group_sparse_stride


count= 1


log_txt_name = dataset + '_iteration_'+str(iteration)+'_gsparse_lamda_' + str(globle_lamda) \
               + '_size_' + str(g_size) + '_step_' + str(g_step)+'_count_'+ str(count) +'.txt'




def weight_variable(name, shape):
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer())
    return v
def bias_variable(name,shape):
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer(),
        regularizer=None, trainable=False, collections=None)
    return v


def inference(images,batch_size):

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    drop_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size,tf.float32)

    outputs = []

    #lt_squar_init = tf.constant(1.0/batch_size,dtype=tf.float32,shape=[batch_size,8,8])

    state = initial_state
    w_lt = weight_variable('w_lt',[hidden_size,64])
    conv_k = weight_variable('conv_k', [1, 1, 2048, 1])
    w_att = weight_variable('w_att', [128, 64])
    w_tanh = weight_variable('w_tanh',[hidden_size, class_num])
    b_tanh = bias_variable('b_tanh',[class_num])


    with tf.variable_scope("RNN"):

        for time_step in range(num_steps):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            last_h = state[-1][-1]

            lt_flat = tf.matmul(last_h,w_lt)
            lt_flat_softmax=tf.nn.softmax(lt_flat)
            #lt_squar=tf.reshape(lt_flat_softmax,[batch_size,8,8])

            Xt = images[:, time_step, :, :, :]
            conv = tf.nn.conv2d(Xt, conv_k, [1, 1, 1, 1], padding='SAME')
            conv = tf.reshape(conv, [-1, 64])
            conv_softmax = tf.nn.softmax(conv)

            if time_step ==0:
                lt_squar = tf.ones(tf.shape(conv_softmax), dtype=tf.float32)
            else:
                lt_squar = tf.reshape(lt_flat_softmax,[-1,8,8])


            conv_flat = tf.contrib.layers.flatten(conv_softmax)
            lt_squar_flat = tf.contrib.layers.flatten(lt_squar)
            merge = tf.concat(1, [lt_squar_flat, conv_flat])

            merge_flat = tf.matmul(merge, w_att)
            merge_softmax = tf.nn.softmax(merge_flat)
            lt_merge = tf.reshape(merge_softmax, [-1, 8, 8])

            lt_merge_expend = tf.expand_dims(lt_merge, 3)
            t_Xt = tf.mul(Xt, lt_merge_expend)

            Xt_flat = tf.reduce_sum(t_Xt,[1,2])

            (cell_output, state) = cell(Xt_flat, state)
            out_after_tanh=tf.tanh(tf.matmul(cell_output, w_tanh) + b_tanh)
            logits = tf.nn.softmax(out_after_tanh)
            #print 'kkk',logits.get_shape()
            outputs.append(logits)

    return outputs


def get_accuracy(rows):
    with tf.Graph().as_default():
        test_images = tf.placeholder(tf.float32, shape=[None, 30, 8, 8, 2048])
        batch_size = tf.placeholder(tf.int32, shape=[])

        out_put = inference(test_images, batch_size)
        predict = predict_for_test(out_put)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, model_dir+model_name)

            t_num=0
            test_accuracy = 0.
            f = open(log_txt_dir+log_txt_name,'a')
            f.write('\n')
            f.close()

            # print "1"

            for filename in rows:
                test_blocks,test_targets = get_test_data(filename)
                # print "2"
                if test_blocks == None:
                    continue
                try:
                    # print "3"
                    prediction = sess.run(predict, feed_dict={test_images: test_blocks, batch_size:len(test_blocks)})
                    # print "4"
                except ZeroDivisionError:
                    continue

                u = 'prediction = %d, target = %f'%(prediction,test_targets)
                print u
                if str(prediction)==str(test_targets):
                    test_accuracy+=1
                t_num+=1
                r = 'test number = %d, immediate = %f'%(t_num,test_accuracy/t_num)
                #print 'test number = ',t_num,' immediate accuracy = ',test_accuracy/t_num
                print r
                file_record = open(log_txt_dir+log_txt_name,'a')
                file_record.write('\n'+u)
                file_record.write('\n'+r)

                file_record.close()

            return test_accuracy,t_num


def predict_for_test(outputs_for_test):
    # tf.get_variable_scope().reuse_variables()
    outputs_for_test = tf.transpose(outputs_for_test,[1,0,2])
    predict_over_timestep = tf.reduce_sum(outputs_for_test,[1])

    predict_over_all_blocks = tf.reduce_sum(predict_over_timestep,[0])
    predict=tf.argmax(predict_over_all_blocks,0)

    return predict


def get_test_data(filename):
    vocab = [_class for _class in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, _class))]
    vocab = sorted(vocab)

    fn = video_path+filename
    label_name = filename.split('/')[0]
    num_label = vocab.index(label_name)

    # to get frame_matrix restoring N*image
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print "could not open : ",fn
        sys.exit()
    ret = True
    frame_matrix=[]
    while(ret):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame,(224,224))
            frame_matrix.append(frame)
    cap.release()
    frame_matrix=np.array(frame_matrix)
    # print "1.1"
    features_matrix=get_features_by_CNN.get_features_matrix(frame_matrix)
    # print "1.2"
    len_of_frames = len(frame_matrix)
    if len_of_frames < 30:
        return None, None

    feature_batch_matrix=[]
    for i in range(0,len_of_frames-30+1,2):
        feature_batch_matrix.append(features_matrix[i:i+30])

    return feature_batch_matrix, num_label


def test():
    f=open(test_sample_txt,'r')
    rows = [row.split()[0] for row in f if row.split()[0] != '']
    accuracy_number = 0.
    total_number = 0.
    for i in range(10):
        _accuracy_number, _total_number = get_accuracy(rows[i*(test_video_num//10):(i+1)*(test_video_num//10)])
        accuracy_number += _accuracy_number
        total_number += _total_number
    file_record = open(log_txt_dir+log_txt_name,'a')
    file_record.write('\n'+'total accuracy for test = '+str(accuracy_number/total_number))
    file_record.close()

if __name__ == "__main__":
    test()
    # # for filename in rows:
    # f = open('shuffled_test_split1_filename.txt', 'r')
    # rows = [row.split()[0] for row in f]
    # filename = rows[0]
    # test_blocks, test_targets = get_test_data(filename)
    # filename = rows[1]
    # test_blocks, test_targets = get_test_data(filename)