import matplotlib.pyplot as plt
from keras.applications import vgg16
model = vgg16.VGG16()
print(model.summary())

from keras.utils import load_img
from keras.utils import img_to_array
from IPython.display import display
def predict_vgg16(model, filename):
    image = load_img(filename)
    display((image))
    image = load_img(filename, target_size=(224, 224))
    plt.imshow(image)
    plt.show()
    # 이미지 데이터를 numpy로 변환
    image = img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = vgg16.preprocess_input(image)
    yhat = model.predict(image)
    label = vgg16.decode_predictions(yhat)
    label = label[0][0]
    # 예측한 라벨과 확률
    print('%s (%.2f%%)'%(label[1], label[2] * 100))

predict_vgg16(model, 'animal/cat7.jpg')
