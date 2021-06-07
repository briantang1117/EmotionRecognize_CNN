# 2020年11月14日
# Brian Tang
# Python 3.8
# tensorflow 2.3.1

from tensorflow import keras
import models_Keras  # 导入模型


class_names = ['anger', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised']
train_folder = 'data/train'
val_folder = 'data/val'
height = 48
width = 48
channels = 1  # 数据集为黑白图像
batch_size = 32
num_classes = 7

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=True,
    class_mode='categorical')

valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)
valid_generator = train_datagen.flow_from_directory(
    val_folder,
    target_size=(height, width),
    batch_size=batch_size,
    seed=7,
    shuffle=False,
    class_mode='categorical')

train_num = train_generator.samples
valid_num = valid_generator.samples
print("训练集数量：" + str(train_num), "验证集数量：" + str(valid_num))

# 建立模型
model = models_Keras.build_model(width, height, num_classes)

# 训练模型
epochs = 100
model.fit(
    train_generator,
    steps_per_epoch=train_num // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num // batch_size
)

# 保存模型
model.save('model_finish.h5')
