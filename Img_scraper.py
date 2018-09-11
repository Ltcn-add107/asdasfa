import urllib.request
import numpy as np
import os
import cv2
#import ImageChops


def store_raw_images():
    #img link
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02969323'
    #neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02131653'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    if not os.path.exists('trainthis'):
        os.makedirs('trainthis')

    pic_num=1
    img_rows=224
    img_cols=224
    img_channels = 3
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "trainthis/"+"road."+str(pic_num)+'.jpg')
            img = cv2.imread("trainthis/"+"road."+str(pic_num)+'.jpg', cv2.IMREAD_COLOR)
            #resized_image = cv2.resize(img, (img_cols,img_rows))
            cv2.imwrite("trainthis/"+"road."+str(pic_num)+'.jpg',img)
            pic_num+=1
        except Exception as e:
            print(str(e))




#def find_uglies():
#    #im1='./ugly/'
#    for file_type in ['neg']:
#        for img in os.listdir(file_type):
#            for ugly in os.listdir('uglies'):
#                try:
#                    current_image_path = str(file_type)+'/'+ str(img)
#                    ugly = cv2.imread('uglies/'+str(ugly))
#                    question = cv2.imread(current_image_path)
#
#                    if not (np.bitwise_xor(ugly,question).any()) and ugly.shape() == question.shape():
#                        #print ('ugly detected!')
#                        #print(current_image_path)
#                        os.remove(current_image_path)
#                except Exception as e:
#                    print(str(e))
#
#

def find_uglies():
    match = False
    for file_type in ['trainthis']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        #print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))



#store_raw_images()
find_uglies()
