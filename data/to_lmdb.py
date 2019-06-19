import os
import lmdb
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import argparse

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:

        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)

def createDataset(outputPath, imagePathList, labelList, map_size, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=map_size)
    # env = lmdb.open(outputPath)
    cache = {}
    cnt = 0
    for i in range(nSamples):
        # print(cnt)
        imagePath = imagePathList[i].replace('\n', '').replace('\r\n', '')
        # print(imagePathList[i])
        # print(imagePath)

        label = labelList[i]
        # print(label)

        # if not os.path.exists(imagePath):
        #     print('%s does not exist' % imagePath)
        #     continue	

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt != 0 and cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    map_size=10000000000
    cur_path=os.getcwd()

    train_save_path=cur_path+'/train_lmdb/'
    train_img_path = cur_path + '/train_images/'
    tain_label=cur_path+'/train.txt'
    train_imgdata = open(tain_label, mode='rb')
    lines = list(train_imgdata)

    imgPathList = []
    labelList = []
    for line in lines:
        imgPath = os.path.join(train_img_path, line.split()[0].decode('utf-8'))
        # print(imgPath)
        imgPathList.append(imgPath)
        word = line.split()[1]
        # print(word)
        labelList.append(word)
    createDataset(train_save_path, imgPathList, labelList, map_size)
    print('------------------------------------------')
    valid_save_path = cur_path + '/valid_lmdb/'
    valid_img_path = cur_path + '/valid_images/'
    valid_label = cur_path + '/valid.txt'
    valid_imgdata = open(valid_label, mode='rb')
    valid_lines = list(valid_imgdata)

    valid_imgPathList = []
    valid_labelList = []
    for valid_line in valid_lines:
        valid_imgPath = os.path.join(valid_img_path, valid_line.split()[0].decode('utf-8'))
        valid_imgPathList.append(valid_imgPath)
        valid_word = valid_line.split()[1]
        valid_labelList.append(valid_word)
    createDataset(valid_save_path, valid_imgPathList, valid_labelList, map_size)