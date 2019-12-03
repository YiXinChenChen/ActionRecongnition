#coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
import os
#import classify_image
import get_features_by_CNN
import shutil
import random
import sys

video_path = '/media/civic/sdc/cyx/Attention/action_youtube_naudio/'
video_file_name = '/media/civic/sdc/cyx/Attention/action_youtube_naudio/ucf11_train_split_2.txt'
tfrecords_path = './tfrecords/'
npy_path = './npys/'

f=open(video_file_name,'r')

rows = [row.split()[0] for row in f if row.split()[0] != '']
row_split = rows[:]

row_split1 = rows[0:300]
row_split2 = rows[300:2*300]
row_split3 = rows[2*300:]


globle_total_size = 0
vocab = [_class for _class in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, _class))]
vocab = sorted(vocab)
#print vocab

def encode_to_tfrecords(row_split, split_number, dataset):


    writer = tf.python_io.TFRecordWriter(tfrecords_path + dataset +'_train_'+ split_number+'.tfrecords')
    
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
    os.mkdir(npy_path)
    number = 0
    for filename in row_split:

        fn=video_path+filename
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



        print fn + "shape:", frame_matrix.shape


        features_matrix=get_features_by_CNN.get_features_matrix(frame_matrix)

        len_of_frames = len(frame_matrix)
        feature_batch_matrix=[]
        for i in range(0,len_of_frames-30+1,2):
            feature_batch_matrix.append(features_matrix[i:i+30])


        f = file(npy_path+str(num_label)+'.npy','ab')
        for i in range(len(feature_batch_matrix)):
            fbm=feature_batch_matrix[i].astype(np.float32)              
            np.save(f,fbm)
        f.close()

        number += 1
        print filename+'has been writen into a npy file. '
        print 'total number = ',number


    name_dict={}
    for i in os.listdir(npy_path):
        name_dict[i]= file(npy_path+i,'rb')

    npy_to_tfrecord_V2(writer)
    shutil.rmtree(npy_path,ignore_errors=1)
    
    writer.close()


def npy_to_tfrecord_V2(writer):
    f ={}
    u=0
    for i in os.listdir(npy_path):
        f[i] = file(npy_path+i,'rb')
    # print f

    name_list,size_list = get_Npy_Size(npy_path)
    nvs = zip(name_list,size_list)
    name_size_dict = dict( (name,value) for name,value in nvs)


    while True:
        # print f.keys()

        i = random_pick(name_size_dict.keys(),name_size_dict.values())
        #print i
        #for i in f.keys():
        try:
            # print i
            num_label=int(i.split('.')[0])
            fbm = np.load(f[i])
            #print fbm.shape
            fbm_raw=fbm.tostring()
            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                            'feature_batch':tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[fbm_raw])
                            ),
                            'label_batch':tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[num_label])
                            )
                        }
                    )
                )
            serialized=example.SerializeToString()
            writer.write(serialized)
        except IOError:
            print i+' runs out'
            del f[i]
            del name_size_dict[i]

        if not f.keys():
            break


def random_pick(some_list, probabilities):  
    x = random.uniform(0,1)  
    cumulative_probability = 0.0  
    for item, item_probability in zip(some_list, probabilities):  
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break  
    return item


def get_Npy_Size(npy_path):
    name_size_dict={}
    total_size = 0.
    for root, dirs, files in os.walk(npy_path):
            for ele in files:
                ele_size = os.path.getsize(root+ele)
                name_size_dict[ele]= ele_size
                total_size += ele_size
            print name_size_dict.items()
            size_list = [name_size_dict[i]/total_size for i in name_size_dict.keys()]
            name_list = name_size_dict.keys()
            print name_list,size_list
    return name_list,size_list


if __name__ == "__main__":
    # encode_to_tfrecords(row_split, 'split2', dataset='ucf11')
    encode_to_tfrecords(row_split1, 'split_11', dataset='ucf11')
    encode_to_tfrecords(row_split2, 'split_22', dataset='ucf11')
    encode_to_tfrecords(row_split3, 'split_33', dataset='ucf11')