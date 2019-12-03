# *- coding: utf-8 -*-
import os, shutil
import numpy as np
import tensorflow as tf
import extrat_features_from_tfrecords


video_path = '/media/civic/sdc/cyx/Attention/action_youtube_naudio/'

"""config."""
class_num = 11
num_layers = 2
num_steps = 30
hidden_size = 512

keep_prob = 0.5
globle_gamma = 1e-5
globle_lamda = 1e-4
lr_rate = 1e-5
max_epoch = 6001
batch_size = 256
group_sparse_size = 4
group_sparse_stride = 4

log_txt_dir = './logs_txt/'
model_dir = './tfs_saver/'

dataset = 'UCF11'

save_name = dataset + '_gsparse' + '_size_'+ str(group_sparse_size) +'_step_'+ str(group_sparse_stride) \
            + '_count_'+ '_lamda_'+str(globle_lamda)  + '_iteration_' + str(max_epoch)

model_name = model_dir + save_name
log_name = log_txt_dir + save_name

if not os.path.exists(log_txt_dir):
    os.mkdir(log_txt_dir)

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def weight_variable(name, shape):
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer())
    return v


def bias_variable(name, shape):
    v = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.random_normal_initializer(),
                        regularizer=None, trainable=False, collections=None)
    return v




def inference(images, batch_size, True_or_False=False):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.5, state_is_tuple=True)
    drop_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    cell = tf.nn.rnn_cell.MultiRNNCell([drop_cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs = []
    lt_out = []
    outputs_for_loss = []

    state = initial_state
    w_lt = weight_variable('w_lt', [hidden_size, 64])
    conv_k = weight_variable('conv_k', [1, 1, 2048, 1])
    # bias_k = bias_variable('bias_k', [1])
    w_att = weight_variable('w_att', [128, 64])
    w_tanh = weight_variable('w_tanh', [hidden_size, class_num])
    b_tanh = bias_variable('b_tanh', [class_num])

    with tf.variable_scope("RNN"):

        for time_step in range(num_steps):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()

            last_h = state[-1][-1]

            lt_flat = tf.matmul(last_h, w_lt)
            lt_flat_softmax = tf.nn.softmax(lt_flat)

            if time_step == 0:
                lt_squar = tf.constant(1.0, dtype=tf.float32, shape=[batch_size, 8, 8])
            else:
                lt_squar = tf.reshape(lt_flat_softmax, [-1, 8, 8])

            Xt = images[:, time_step, :, :, :]

            conv = tf.nn.conv2d(Xt, conv_k, [1, 1, 1, 1], padding='SAME')
            conv = tf.reshape(conv, [-1, 64])
            conv_softmax = tf.nn.softmax(conv)
            # bias = tf.nn.bias_add(conv, bias_k)
            # conv = tf.nn.relu(bias)

            conv_flat = tf.contrib.layers.flatten(conv_softmax)
            lt_squar_flat = tf.contrib.layers.flatten(lt_squar)
            merge =  tf.concat(1, [lt_squar_flat, conv_flat])

            merge_flat = tf.matmul(merge, w_att)
            merge_softmax = tf.nn.softmax(merge_flat)
            lt_merge = tf.reshape(merge_softmax, [-1, 8, 8])


            lt_merge_expend = tf.expand_dims(lt_merge, 3)
            t_Xt = tf.mul(Xt, lt_merge_expend)


            Xt_flat = tf.reduce_sum(t_Xt, [1, 2])

            (cell_output, state) = cell(Xt_flat, state)
            out_after_tanh = tf.tanh(tf.matmul(cell_output, w_tanh) + b_tanh)
            logits = tf.nn.softmax(out_after_tanh)

            outputs.append(logits)
            lt_out.append(lt_merge_expend)
            outputs_for_loss.append(out_after_tanh)
        lt_out_tensor = tf.concat(3, lt_out)
        trainable_variables = tf.trainable_variables()

    return outputs, trainable_variables, lt_out_tensor, outputs_for_loss

def loss_att(lt_out_tensor, k_s, stride):
    [_, __, ___, chanel] = lt_out_tensor.get_shape().as_list()

    kernel  = tf.constant(1.0, dtype=tf.float32, shape=[k_s, k_s, chanel, 1])
    lt_out_tensor = tf.square(lt_out_tensor)
    lt_out_loss = tf.nn.depthwise_conv2d(lt_out_tensor, kernel, [1, stride, stride, 1], padding='VALID')
    lt_out_loss = tf.sqrt(lt_out_loss)
    # print lt_out_loss.get_shape().as_list()
    loss  = tf.reduce_sum(lt_out_loss)
    return loss


def loss_v2(outputs_for_loss, labels, lt_out_loss, trainable_variables):
    cross_entropy = 0.
    for i in range(num_steps):
        cross_entropy += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs_for_loss[:, i, :],
                                                                        labels=labels[:, i])

    final = 0.
    for i in trainable_variables:
        final += tf.nn.l2_loss(i)
    final_lost = globle_gamma * final + cross_entropy/30.0 + globle_lamda * lt_out_loss/30.0

    return tf.reduce_mean(final_lost)


def evaluation_for_train(outputs, labels):
    argmax_on_timestep = tf.reduce_sum(outputs, 1)  # 64*51
    predict = tf.argmax(argmax_on_timestep, 1)  # 64
    correct_prediction = tf.equal(predict, labels[:, 1])
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(correct_prediction)


def training(lost):
    train_step = tf.train.AdamOptimizer(lr_rate).minimize(lost)
    return train_step


def train(batch_size):
    with tf.Graph().as_default():

        images = tf.placeholder(tf.float32, shape=[None, 30, 8, 8, 2048])
        labels = tf.placeholder(tf.int64, shape=[None, 30])


        images_tfrecord, labels_tfrecord = extrat_features_from_tfrecords.generate_feature_batch_and_label_batch(
            batch_size)
        # images = 64*30*8*8*2048   labels = 64*30

        outputs, trainable_variables, lt_out, outputs_for_loss = inference(images, batch_size)

        # outputs = 30*64*51  lt_out = 30*64*8*8
        outputs = tf.transpose(outputs, [1, 0, 2])
        att_loss = loss_att(lt_out, group_sparse_size, group_sparse_stride)
        outputs_for_loss = tf.transpose(outputs_for_loss, [1, 0, 2])

        # predict=tf.argmax(outputs,2)

        train_accuracy = evaluation_for_train(outputs, labels)

        lost = loss_v2(outputs_for_loss, labels, att_loss, trainable_variables)
        # lost = (64,)
        train_op = training(lost)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            for i in range(max_epoch):  # 10 times total epoch

                images_tmp, labels_tmp = sess.run([images_tfrecord, labels_tfrecord])

                _, out_lost = sess.run([train_op, lost], feed_dict={images: images_tmp, labels: labels_tmp})
                # print type(out_lr)
                if i % 10 == 0:
                    # r = 'time_step %d, lost = %g, at_lost = %g' % (i, out_lost, att_lost)
                    r = 'time_step %d, lost = %g' % (i, out_lost)
                    print r
                if i % 50 == 0:
                    out_accuracy = sess.run(train_accuracy, feed_dict={images: images_tmp, labels: labels_tmp})
                    u = 'time_step %d, accuracy = %g' % (i, out_accuracy)
                    print u
                    # if i%500==0:
                    file_record = open(log_name + '.txt', 'a')
                    file_record.write('\n' + r)
                    file_record.write('\n' + u)
                    file_record.close()

            coord.request_stop()
            coord.join(threads)
            save_path = saver.save(sess, model_name + '.ckpt')

if __name__ == "__main__":
    train(batch_size)
    # aa = np.random.normal(size=(64, 30, 8, 8))
    # handel_weight(aa, 4)