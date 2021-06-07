# 2020年11月14日
# Brian Tang
# Python 3.8
# tensorflow 2.3.1

from tensorflow import keras


# 建立模型
def build_model(width, height, num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu',
                                  input_shape=(width, height, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=16, kernel_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=32, kernel_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2))
    model.add(keras.layers.Conv2D(filters=7, kernel_size=3, strides=1, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.4))  # Dropout防止过度拟合
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )
    model.summary()
    return model
