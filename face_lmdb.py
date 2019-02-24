#coding=utf-8

'''
通过facenet得到的512特征写入lmdb文件中
'''

import  lmdb
import  os
import  face_comm

class face_lmdb:
    def add_embed_to_lmdb(self,id,name,vector):
        self.db_file=os.path.abspath(face_comm.get_conf('lmdb','lmdb_path'))
        id = str(id)+','+str(name)
        evn = lmdb.open(self.db_file)
        wfp = evn.begin(write=True)
	    # print vector
	    # print face_comm.embed_to_str(vector)
        wfp.put(key=id, value=face_comm.embed_to_str(vector))
        wfp.commit()
        evn.close()

    def load_index_from_lmdb(self,id_list,name_list):
        # 遍历
        self.db_file=os.path.abspath(face_comm.get_conf('lmdb','lmdb_path'))
        if os.path.isdir(self.db_file):
            evn = lmdb.open(self.db_file)
            wfp = evn.begin()

            for key, value in wfp.cursor():
                str_list = key.split(',')
                id_list.append(str_list[0])
                name_list.append(str_list[1])
                # print 'lmdb-id=',str_list[0],'lmdb-name=',str_list[1]

    def show_lmdb(self):
        self.db_file=os.path.abspath(face_comm.get_conf('lmdb','lmdb_path'))
        evn = lmdb.open(self.db_file)
        wfp = evn.begin()
       # for key,value in wfp.cursor():
       #     print key

if __name__=='__main__':
    #插入数据
    embed = face_lmdb()
    embed.show_lmdb()
