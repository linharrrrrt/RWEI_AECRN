import os
import cv2
import numpy as np
import math

# you need change this root path to decompressed ecd datasets path
root_path  = "shapes_6dof/"

def mkdir(path):
 
    folder = os.path.exists(path)
 
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def patchNZGEentropy(patch):
    count = np.sum(patch)
    entropy = (count/16) *math.log((count/16),2)
    return -1*entropy

def quaternion2rotation(quat):
    # quat:(w,x,y,z)
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat
 
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d
 
    # s = a2 + b2 + c2 + d2
 
    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2
 
    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

def q2m(x,y,z,qx,qy,qz,qw): #四元数位姿转矩阵位姿
    
    rMatrix = quaternion2rotation((qw,qx,qy,qz))
    posematrix = np.zeros((4,4), dtype  =  np.float32)
    posematrix[3,3]=1
    posematrix[0:3,0:3]=rMatrix
    posematrix[0,3]=x
    posematrix[1,3]=y
    posematrix[2,3]=z
    return posematrix 

def readeventtoRWE():# RWEI from groundtruth
    
    mkdir(root_path+"poses_rwei")
    Threshold_low = 0.35 # 具体阈值由信息熵的底不同而设定
    savepath = root_path+"eventimages_rwei_"+str(Threshold_low)+"/"
    mkdir(savepath)
    event_shape = (180,240)
    eventwindow = [] #event split window
    windowsize = event_shape[0]*event_shape[1]*1.5 # windows size
    with open(root_path+"groundtruth.txt","r") as images: # Timestamp got from this file, pay attention.
        imageslists = images.readlines()
        with open(root_path+"events.txt","r") as events:
            mycount = 0
            for index,imageitem in enumerate(imageslists):
                mycount+=1
                imageitem = imageitem.split(" ")
                imagetmstamp = float(imageitem[0])
                # initialization
                EventImage_all = np.zeros((event_shape[0],event_shape[1],3),dtype=float)
                EventImagep = np.zeros((event_shape[0],event_shape[1]),dtype=float)
                EventImagen = np.zeros((event_shape[0],event_shape[1]),dtype=float)
                pqp = np.zeros((event_shape[0]//4,event_shape[1]//4),dtype=float)
                pqn = np.zeros((event_shape[0]//4,event_shape[1]//4),dtype=float)
                NZGEp = 0
                NZGEn = 0
                count = 0 #辅助计量，因为要判断一下是不是这个位姿能生成帧
                # img[100][200]=[0,0,255]#red
                # img[50][100]=[0,255,0]#green
                # img[60][200]=[255,0,0]#blue
                ngridp = 1
                ngridn = 1
                pqnsum = 0
                pqpsum = 0
                while(True):
                    event = events.readline()
                    event = event.split(" ")
                    eventwindow.append(event)
                    if len(eventwindow)>windowsize:
                        eventwindow.pop(0)
                    eventtmstamp = float(event[0])
                    if eventtmstamp>imagetmstamp:
                        for i in range(len(eventwindow)-1,0,-1):
                            # 所以这个地方要记录一下开始生成帧是从什么位置开始的。每次都从这个下标开始。初始下标就是从最后一个开始。
                            count += 1
                            event = eventwindow[i]
                            event[1] = int(event[1])
                            event[2] = int(event[2])
                            event[3] = int(event[3])
                            
                            if int(event[3])==0:
                                EventImagen[event[2]][event[1]]=1
                                # 更快速的熵处理方式
                                x = event[2]//4 
                                y = event[1]//4
                                # 更新部分熵，
                                if pqn[x][y]>0.0:
                                    pass
                                else:
                                    ngridn +=1
                                earlypqnxy = pqn[x][y]
                                pqn[x][y] = patchNZGEentropy(EventImagen[x*4:(x+1)*4,y*4:(y+1)*4])
                                pqnsum = pqnsum + pqn[x][y] - earlypqnxy
                                NZGEn = pqnsum/ngridn
                            # 根据极性进行帧赋值，
                            if event[3]==1:
                                EventImagep[event[2]][event[1]]=1
                                # 更快速的熵处理方式
                                x = event[2]//4 
                                y = event[1]//4
                                # 更新部分熵，
                                if pqp[x][y]>0.0:
                                    pass
                                else:
                                    ngridp +=1
                                earlypqpxy = pqp[x][y]
                                pqp[x][y]=patchNZGEentropy(EventImagep[x*4:(x+1)*4,y*4:(y+1)*4])
                                pqpsum = pqpsum + pqp[x][y] - earlypqpxy
                                NZGEp = pqpsum/ngridp
                            # 重新计算全局熵，

                            entropy = (NZGEn+NZGEp)/2
                            # 判断是否达到阈值，如果达到阈值，
                            if entropy>Threshold_low:
                                # generate image
                                EventImage_all[:,:,2] = np.uint8(EventImagep*255)
                                EventImage_all[:,:,1] = np.uint8(EventImagen*255)
                                # filename = "EventImage_"+str(poseindex).zfill(8)+".png"# 务必使用png，使用jpg压缩之后，会使得原来的值发生变化
                                cv2.imwrite(savepath+"/event_image_rwei_"+str(index).zfill(8)+".png", EventImage_all) # must be png image, jpg file will compress the image information
                                print(savepath+"/event_image_rwei_"+str(index).zfill(8)+".png")
                                posematrix = q2m(float(imageitem[1]),float(imageitem[2]),float(imageitem[3]),float(imageitem[4]),float(imageitem[5]),float(imageitem[6]),float(imageitem[7]))#(x,y,z,qx,qy,qz,qw)
                                np.savetxt(root_path+"poses_rwei/pose_"+str(index).zfill(8)+".txt",posematrix)
                                # 退出事件遍历
                                break
                        break
readeventtoRWE()