import tensorflow as tf
from model.base_model_provider import AbstractModelProvider

class UNetModel2(AbstractModelProvider):
    
    def down_block(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = tf.keras.layers.BatchNormalization()(c)
        p = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        p = tf.keras.layers.Dropout(0.3)(p)
        return c, p

    def up_block(self, x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = tf.keras.layers.UpSampling2D((2, 2))(x)
        concat = tf.keras.layers.Concatenate()([us, skip])
        c = tf.keras.layers.Dropout(0.3)(concat)
        c = tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = tf.keras.layers.BatchNormalization()(c)
        return c

    def bottleneck(self, x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = tf.keras.layers.BatchNormalization()(c)
        return c

    def get_model(self, img_size, num_classes) -> tf.keras.Model:
        f = [16, 32, 64, 128, 256]
        inputs = tf.keras.layers.Input(shape=img_size + (3,))
        
        p0 = inputs
        c1, p1 = self.down_block(p0, f[0]) #128 -> 64
        c2, p2 = self.down_block(p1, f[1]) #64 -> 32
        c3, p3 = self.down_block(p2, f[2]) #32 -> 16
        c4, p4 = self.down_block(p3, f[3]) #16->8
        
        bn = self.bottleneck(p4, f[4])
        
        u1 = self.up_block(bn, c4, f[3]) #8 -> 16
        u2 = self.up_block(u1, c3, f[2]) #16 -> 32
        u3 = self.up_block(u2, c2, f[1]) #32 -> 64
        u4 = self.up_block(u3, c1, f[0]) #64 -> 128
        
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(u4)
        # outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(u4)
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model