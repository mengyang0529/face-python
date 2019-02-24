#coding=utf-8

import face_detect

import  face_annoy
import  face_alignment
import  face_encoder
import  face_lmdb
import  cv2
detect = face_detect.Detect()
encoder = face_encoder.Encoder()
annoy = face_annoy.face_annoy()

#获得对齐人脸图片
def get_align_pic(pic):
    result = detect.detect_face(pic)
    if len(result['boxes']):
        align = face_alignment.Alignment()
        return  align.align_face(pic, result['face_key_point'])
    else:
        return None

#计算人脸特征
def get_face_embed_vector(align_pic):
    return encoder.generate_embedding(align_pic)



#添加图片索引
def add_face_index(id,name,pic):
    align_face = get_align_pic(pic)
    if align_face is not None:
        #获取人脸特征
        face_vector = get_face_embed_vector(align_face)
	print type(face_vector)
        # 插入数据
        embed = face_lmdb.face_lmdb()
        embed.add_embed_to_lmdb(id,name,face_vector)
        #更新索引
        annoy.create_index_from_lmdb()
        annoy.reload()
        print "add user:" +str(id)+ " from " + pic
        return True
    else:
        print "none face"
        return False


def query_face(pic):
    align_face = get_align_pic(pic)
    if align_face is not None:
        #获取人脸特征
        face_vector = get_face_embed_vector(align_face)
        return annoy.query_vector(face_vector)

def detect_face(pic):
    result = detect.detect_face(pic)
    return result

if __name__=='__main__':

    pic='web/images/qxr.jpeg'
    add_face_index(0,"rui",pic)

    pic2='web/images/0000.jpeg'
    add_face_index(1,"yali",pic2)

    #print query_face(pic)
    #print detect_face(pic)
