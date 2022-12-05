import tensorflow as tf
from model.base_model_provider import AbstractModelProvider

class UNetModel(AbstractModelProvider):
    
    def get_model(self, img_size, num_classes) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=img_size + (3,))
        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = tf.keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.UpSampling2D(2)(x)

            # Project residual
            residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
            residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
            x = tf.keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = tf.keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=[tf.keras.metrics.Accuracy()])
        return model
    