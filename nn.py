from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Model

def defin_model(input_shape, n_classes):
    assert isinstance(input_shape, list) or isinstance(input_shape, tuple)
    assert len(input_shape) == 3

    x0 = Input(input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x0)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)

    preds = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=x0, outputs=preds)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
