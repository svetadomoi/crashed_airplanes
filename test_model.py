import tensorflow as tf
import cv2
import pandas as pd
from model import ResNetLike

def read_image(directory, img):
    IMG_SIZE = 20
    img_array = cv2.imread(directory+img, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE,3)


model = ResNetLike()
model.build()
model.model.load_weights('w.h5')

df = pd.read_csv('test/test.csv')
imgs = df['filename'].values
labels = df['sign'].values

cnt = 0

for i in range(len(imgs)):
    prediction = int(model.model.predict(read_image('test/',imgs[i]))[0][0])
    if prediction == labels[i]:
        cnt += 1

print(cnt/len(labels))
