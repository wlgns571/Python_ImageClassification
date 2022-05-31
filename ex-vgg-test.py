from keras.models import load_model
from keras.applications import vgg16
from keras.utils import load_img
from keras.utils import img_to_array
from IPython.display import display
import matplotlib.pyplot as plt

model = load_model('./model/vgg16_first.h5')
print(model.summary())

from week4.ex_cnn.ex_vgg_image import data_load
train_gen, test_gen, class_num, class_label = data_load()
import numpy as np
def predict_vgg16(model, filename):
    image = load_img(filename)
    display(image)
    image = load_img(filename, target_size=(224, 224))
    plt.imshow(image)
    plt.show()
    # 이미지 데이터를 numpy 로변환
    image = img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    yhat = model.predict(image)
    # 최대 인덱스
    idx = np.argmax(yhat[0])
    # 예측 레이블
    print('%s (%.2f%%)' %(class_label[idx], yhat[0][idx]*100))

predict_vgg16(model, './dental_image/test/decayed/103.jpg')