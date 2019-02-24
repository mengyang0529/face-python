#coding=utf-8
import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import face_net.src.facenet as facenet
import face_net.src.align.detect_face
import  face_annoy
import  face_comm
import  face_lmdb
import time

np.set_printoptions(suppress=True)
gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.abspath(face_comm.get_conf('facedetect','model'))

annoy = face_annoy.face_annoy()

class face_annoy:

    def __init__(self):
        self.f                = int(face_comm.get_conf('annoy','face_vector'))
        self.annoy_index_path = os.path.abspath(face_comm.get_conf('annoy','index_path'))
        self.lmdb_file        =os.path.abspath(face_comm.get_conf('lmdb','lmdb_path'))
        self.num_trees        =int(face_comm.get_conf('annoy','num_trees'))

        self.annoy = AnnoyIndex(self.f)
        if os.path.isfile(self.annoy_index_path):
            self.annoy.load(self.annoy_index_path)

    #从lmdb文件中建立annoy索引
    def create_index_from_lmdb(self,id):
        # 遍历
        lmdb_file = self.lmdb_file
        if os.path.isdir(lmdb_file):
            evn = lmdb.open(lmdb_file)
            wfp = evn.begin()
            annoy = AnnoyIndex(self.f)
        
            for key, value in wfp.cursor():
                # key = (key)
                print key
                value = face_comm.str_to_embed(value)
                annoy.add_item(id,value)

            annoy.build(self.num_trees)
            annoy.save(self.annoy_index_path)

    #重新加载索引
    def reload(self):
        self.annoy.unload()
        self.annoy.load(self.annoy_index_path)

    #根据人脸特征找到相似的
    def query_vector(self,face_vector):
        n=int(face_comm.get_conf('annoy','num_nn_nearst'))
        return self.annoy.get_nns_by_vector(face_vector,n,include_distances=True)

class Encoder:
    def __init__(self):
        self.dectection= Detection()
        self.sess = tf.Session()
        start=time.time()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)
        print 'Model loading finised,cost: %ds'%((time.time()-start))

    def generate_embedding(self, image):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
      
        faces=self.dectection.find_faces(image)
        for face in faces:
            prewhiten_face = facenet.prewhiten(face.image)

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
            face.embedding=self.sess.run(embedding, feed_dict=feed_dict)[0]
	    # print type(face.embedding)
        return faces

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            gpu_options = tf.GPUOptions(allow_growth = True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return face_net.src.align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        # image = misc.imread(os.path.expanduser(image), mode='RGB')
        bounding_boxes, _ = face_net.src.align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)
        return faces

if __name__=='__main__':

    cap = cv2.VideoCapture(1)
    encoder = Encoder()
    frameNo=0
    name_list =[]
    id_list=[]
    embed = face_lmdb.face_lmdb()   
    embed.load_index_from_lmdb(id_list,name_list)
    embed.show_lmdb()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces=encoder.generate_embedding(rgb)
        for face in faces:
            if annoy.query_vector(face.embedding)[1][0]<0.8 :
                font = cv2.FONT_HERSHEY_SIMPLEX
                index=0
                found=0
                for id in id_list:
                    id_annoy=int(annoy.query_vector(face.embedding)[0][0])
                    id =int(id)
                    # print id ,id_annoy
                    if id==id_annoy:
                        found=1
                        break
                    else:
                        index=index+1     
                    # print 'frameNo=',frameNo, 'index=',index,'id=',id,'found=',found,'id_annoy=',id_annoy
                if found==1:
                    found=0
                    cv2.putText(frame,str(name_list[index]),(face.bounding_box[0],face.bounding_box[1]), font, 1,(255,255,255),2,1)
                cv2.rectangle(frame,(int(face.bounding_box[0]),int(face.bounding_box[1])),(int(face.bounding_box[2]),int(face.bounding_box[3])),(0,255,0),3)
            else :
                cv2.rectangle(frame,(int(face.bounding_box[0]),int(face.bounding_box[1])),(int(face.bounding_box[2]),int(face.bounding_box[3])),(0,0,255),3)

            # print annoy.query_vector(face.embedding)[0][0] ,annoy.query_vector(face.embedding)[1][0]
        # Display the resulting frame
        frameNo=frameNo+1
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            if(len(faces)==1):
                name = raw_input("Input your name:")
                # id = input("Set your ID:")
                id = len(name_list)
                id_list.append(id)
                name_list.append(name)
                embed.add_embed_to_lmdb(id,name,faces[0].embedding)
                annoy.create_index_from_lmdb()
                annoy.reload()
                embed.show_lmdb()
            else:
                print "Need only one face to regist."
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
