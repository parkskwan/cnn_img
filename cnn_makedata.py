#!/usr/bin/env python3.6

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split


caltech_dir = "./image/101_ObjectCategories"
categories = ["brain", "BACKGROUND_Google", "crayfish", "helicopter", "okapi"]


nb_classes = len(categories)


image_w = 64; image_h = 64 


X = []; Y = []

for idx, f in enumerate(categories):

    label = [ 0 for i in range(nb_classes) ]
    label[idx] = 1

    image_dir = caltech_dir + "/" + f
    files = glob.glob(image_dir + "/*.jpg")

    for i, fname in enumerate(files):

        img = Image.open(fname)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))

        data = np.asarray(img)

        X.append(data); Y.append(label)

        # 각도를 조금 변경한 파일 추가하기
        # 회전하기

        for ang in range(-20, 20, 5):

            img2 = img.rotate(ang); data = np.asarray(img2)
            X.append(data); Y.append(label)


            # 반전하기
            
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT); data = np.asarray(img2)
            X.append(data); Y.append(label)


        if i % 10 == 0:
            print(i, "\n", data)






X = np.array(X); Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

XY = (X_train, X_test, Y_train, Y_test)

np.save("./object/obj_5.npy", XY)

print("ok", len(Y))


