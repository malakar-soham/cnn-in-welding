import numpy as np
import cv2
import os
import random
import tensorflow as tf

h,w = 512,512

def create_model():

    inputs = tf.keras.layers.Input(shape=(h,w,3))

    conv1 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPool2D()(conv1)

    conv2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPool2D()(conv2)

    conv3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPool2D()(conv3)

    conv4 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(pool3)

    upsm5 = tf.keras.layers.UpSampling2D()(conv4)
    upad5 = tf.keras.layers.Add()([conv3,upsm5])
    conv5 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same')(upad5)

    upsm6 = tf.keras.layers.UpSampling2D()(conv5)
    upad6 = tf.keras.layers.Add()([conv2,upsm6])
    conv6 = tf.keras.layers.Conv2D(16,(3,3),activation='relu',padding='same')(upad6)

    upsm7 = tf.keras.layers.UpSampling2D()(conv6)
    upad7 = tf.keras.layers.Add()([conv1,upsm7])
    conv7 = tf.keras.layers.Conv2D(1,(3,3),activation='relu',padding='same')(upad7)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv7)

    return model

images = []
labels = []

files = os.listdir('./dataset/images/')
random.shuffle(files)

for f in files:
    img = cv2.imread('./dataset/images/' + f)
    parts = f.split('_')
    label_name = './dataset/labels/' + 'W0002_' + parts[1]
    label = cv2.imread(label_name,2)

    img = cv2.resize(img,(w,h))
    label = cv2.resize(label,(w,h))

    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)
labels = np.reshape(labels,
    (labels.shape[0],labels.shape[1],labels.shape[2],1))

print(images.shape)
print(labels.shape)

images = images/255
labels = labels/255

model = tf.keras.models.load_model('my_model')

#model = create_model()  # uncomment this to create a new model
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(images,labels,epochs=100,batch_size=10)
model.evaluate(images,labels)

model.save('my_model')
