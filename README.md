# Similar-image-detection

## 비슷한 이미지 검색 시스템 만들기

이미지 파일을 검색하는 경우 생각해 봅니다. 이미지 파일은 크기 조정, 색상 보정 등에 따라 파일크기와 바이너리 데이터가 완전히 바뀌어 버립니다. 따라서 단순한 검색으로는 유사한 이미지를 찾을 수 없습니다. 그럼 어떻게 유사한 이미지를 찾을 수 있을까요? 이번 장에서는 Caltech에서 제공하는 예제를 이용하여 이에 대해 알아 보겠습니다.


## . 학습할 내용:

합성곱(CNN)의 딥러닝 사용해 보기, 서로 다른 크기의 색상이 있는 이미지 분류해보기, TensorFlow + Keras

다운로드(Caltech 101): http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz


## 1. CNN으로 색상이 있는 이미지 분류의 필요성:

인터넷이 급속히 발달하는 가운데 스마트폰, 디지털 카메라, 블랙박스 등의 기기에서 수집되는 방대한 씁상 데이터가 소셜 미디어 사이트를 통해 빠르게 공유되고 있다. 소셜 미디어 공유 사이트에서는 일반적으로 이미지의 태그 정보를 사용하는데, 멀티미디어를 공유하는 방법이 쉬워지고 그 양이 폭발적으로 증가함에 따라 이미지에 태그를 붙여야 하는 일은 번거로움이 되고 있다. 또한 태그가 잘못 붙여지거나 안 붙은 경우에는 이미지 검색 정확도가 떨어질 가능성이 있다. 

Caltech 101에서 제공하는 색상이 있는 이미지 데이터를 CNN(Convolution Neural Network) 딥러닝 기법으로 분류해 보겠습니다.  Caltech 101에는 101가지 종류의 카테고리로 분류된 이미지가 들어 있습니다. 전부  분류하고 학습하는데는 시간이 오래 걸리므로 임의로 5가지 종류의 카테고리를 학습시키고 정확하게 분류할 수 있는지 테스트해 보겠습니다.

임의로 선택한 "brain", "BACKGROUND_Google", "crayfish", "helicopter", "okapi" 카테고리의 서로 다른 크기의 색상 이미지를 사용하겠습니다. 각 카테고리에는 이미지가 각각 98, 467, 70, 88, 39장 정도 있으며, 전체 762장의 사진을 대상으로 분류해보겠습니다.


## 2. 서로 다른 크기의 색상 이미지 데이터를 동일한 규격으로 변환하기: 

Caltech 101의 이미지는 크기가 모두 달라서 머신러닝(딥러닝)에서 다루기에 불편한 면이 있습니다. 따라서 이미지를 일정한 규격으로 리사이즈하고, 
24비트 RGB 64x64 픽셀 형식(실제는 임의로)으로 변환해서 저장 해 놓습니다. 그 이유는 미루어 짐작 할 수 있을 거라 믿습니다.

```python

from PIL import Image
import os, glob
import numpy as np

from sklearn.model_selection import train_test_split


caltech_dir = "./image/101_ObjectCategories"
categories = ["brain", "BACKGROUND_Google", "crayfish", "helicopter", "okapi"]

nb_classes = len(categories)

X = []; Y = []
image_w = 64; image_h = 64 


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

X = np.array(X); Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

XY_val = (X_train, X_test, Y_train, Y_test)

np.save("./object/obj_5.npy", XY_val)
```    




## 3. CNN으로 학습해보기:

일반적인 합성곱 신경망(CNN)으로 학습시켜봅니다. TensorFlow + Keras 조합으로 CNN을 테스트해봅니다.
어느 정도의 정확도가  나올까요? 대략 accuracy = 0.87(87%)의 정도의 정확도가 나오게 됩니다. 실망하지 마세요. 
바로 밑에 있는 섹션(4. 판정 정밀도 올리기)를 참고 하시면 만족스러운 정확도(accuracy = 0.993(99%))를 얻으실수 있습니다. 

```python

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

X_train = X_train.astype("float") / 255; X_test = X_test.astype("float") / 255 


nfilter = bsize = 32; opt = ['adam','rmsprop']


model = Sequential()

model.add(Conv2D(nfilter, (3, 3), padding="same", input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(2*nfilter, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(2*nfilter, (3, 3)))
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



score = model.evaluate(X_test, Y_test, batch_size=bsize)
print("\n\n\n\nloss =", score[0], ", accuracy =", score[1],", baseline error: %.2f%%" % (100-score[1]*100))
```



## 4. 판정 정밀도 올리기:

정밀도를 올릴 수 있게 조금 수정해봅니다. 일단 이미지의 각도를 변경하거나 반전해서 데이터의 수를 늘려봅니다. 
이제 전체 영상이 762장에서 12954장으로 늘어나게 됩니다. 대략 accuracy = 0.993(99%)의 정도의 정확도가 나오게 됩니다.

이미지의 수를 늘릴 때 활용할 수 있는 PIL(Image) 메서드는 다음과 같습니다.

Image.transpose(v) : 90도 단위로 이미지를 회전하거나 반전합니다.

Image.rotate(angle) : 이미지를 angle도 만큼 회전합니다.

```python

from PIL import Image
import os, glob
import numpy as np

from sklearn.model_selection import train_test_split


caltech_dir = "./image/101_ObjectCategories"
categories = ["brain", "BACKGROUND_Google", "crayfish", "helicopter", "okapi"]

nb_classes = len(categories)

X = []; Y = []
image_w = 64; image_h = 64 

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

###############################################################################
###############################################################################

    # 각도를 조금 변경한 파일 추가하기
    # 회전하기

    for ang in range(-20, 20, 5): 

      img2 = img.rotate(ang); data = np.asarray(img2)
      X.append(data); Y.append(label)


      # 반전하기

      img2 = img2.transpose(Image.FLIP_LEFT_RIGHT); data = np.asarray(img2)
      X.append(data); Y.append(label)

###############################################################################
###############################################################################

X = np.array(X); Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

XY_val = (X_train, X_test, Y_train, Y_test)

np.save("./object/obj_5.npy", XY_val)
```    

## 5. 내용정리

....


