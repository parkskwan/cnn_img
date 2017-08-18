#!/usr/bin/env python3.6

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import h5py
from PIL import Image
import numpy as np

categories = ["brain", "BACKGROUND_Google", "crayfish", "helicopter", "okapi"]


nb_classes = len(categories)

image_w = 64; image_h = 64

X_train, X_test, Y_train, Y_test = np.load("./object/obj_5.npy")



X_train = X_train.astype("float") / 256; X_test = X_test.astype("float") / 256

print('X_train shape: ', X_train.shape) 

bsize = 32; opt = ['adam','rmsprop']

model = Sequential()

model.add(Conv2D(bsize, (3, 3), padding="same", input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(2*bsize, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(2*bsize, (3, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy', optimizer=opt[1], metrics=['accuracy'])





hdf5_file="./object/obj_5-model.hdf5"

if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
else:
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=bsize, epochs=5)
    
    model.save_weights(hdf5_file)







pre = model.predict(X_test)

for i, v in enumerate(pre):

    pre_ans = v.argmax()
    ans = Y_test[i].argmax()
    dat = X_test[i]

    if ans == pre_ans: continue

    print("[Not Good]", categories[pre_ans], "!=", categories[ans]) 
    print(v)

    fname = "./image/error/" + str(i) + "-" + categories[pre_ans] + "-ne-" + categories[ans] + ".png"

    dat *= 256

    img = Image.fromarray(np.uint8(dat))
    img.save(fname)



score = model.evaluate(X_test, Y_test, batch_size=bsize)


print()
print()
print()
print()
print("loss =", score[0], ", accuracy =", score[1],", baseline error: %.2f%%" % (100-score[1]*100))


