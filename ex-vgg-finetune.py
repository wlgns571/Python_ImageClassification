from keras import models
from keras import layers
from keras import optimizers
import keras.backend as K
from keras.applications import VGG16
K.clear_session()
conv_layer = VGG16(weights='imagenet'
                   ,include_top=False
                   ,input_shape=(224,224,3))
print(conv_layer.summary())
#conv 학습되지 않도록 고정
for layer in conv_layer.layers:
    layer.trainable = False
# 새로운 모델 생성
model = models.Sequential()
model.add(conv_layer)
model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3,activation='softmax'))
print(model.summary())
from keras.models import load_model

model.compile(loss='categorical_crossentropy'
              ,optimizer=optimizers.RMSprop(lr=1e-4)
              ,metrics=['acc'])
from week4.ex_cnn.ex_vgg_image import data_load
train_gen, test_gen, class_num, class_label = data_load()
history = model.fit_generator(
      train_gen
    , steps_per_epoch=train_gen.samples / train_gen.batch_size
    , epochs=100
    , validation_data=test_gen
    , validation_steps=test_gen.samples / test_gen.batch_size
    , verbose=1
)
model.save('vgg16_first.h5')
