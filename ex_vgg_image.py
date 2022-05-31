from keras.preprocessing.image import ImageDataGenerator

train_dir = './dental_image/train'
test_dir = './dental_image/test'
batch_size = 32
image_size = 224

def data_load():
    train_gen = ImageDataGenerator(
        rotation_range=180       # 회전 최대 180도
        ,width_shift_range=0.2   # 좌우 이동
        ,height_shift_range=0.2  # 상하 이동
        ,horizontal_flip=True    # 좌우 반전
        ,vertical_flip=True      # 상하 반전
    )

    test_gen = ImageDataGenerator()
    train_generator = train_gen.flow_from_directory(
        train_dir
        ,target_size=(image_size, image_size)
        ,batch_size=batch_size
        ,class_mode='categorical'
        ,shuffle=True
    )
    test_generator = test_gen.flow_from_directory(
        test_dir
        ,target_size=(image_size, image_size)
        ,batch_size=batch_size
        ,class_mode='categorical'
        ,shuffle=False
    )
    class_num = len(train_generator.class_indices)
    custom_labels = list(test_generator.class_indices.keys())
    print('class_num',class_num)
    print('custom_labels',custom_labels)

    return train_generator, test_generator, class_num, custom_labels