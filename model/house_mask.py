import tensorflow as tf
import numpy as np
from model.config import Config
from model.data_provider import DataProvider
from model.base_model_provider import AbstractModelProvider

class HouseMask(object):

    model : tf.keras.Model

    def __init__(self, config : Config, train_data_provider : DataProvider, validation_data_provider : DataProvider, model_provider : AbstractModelProvider):
        self.config = config
        self.train_data_provider = train_data_provider
        self.validation_data_provider = validation_data_provider
        self.model_provider = model_provider
        if(model_provider != None):
            self.model = self.model_provider.get_model(self.config.IMAGE_SIZE, self.config.NUM_CLASSES)
    
    def train(self) -> tf.keras.Model:
        monitor = 'val_accuracy' if self.validation_data_provider != None else 'accuracy'
        
        checkpoint_filepath = 'data/checkpoint/hm-{epoch:02d}-{' + monitor + ':.2f}.h5'
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor=monitor,
            mode='max',
            save_best_only=True)
        
        self.model.fit(self.train_data_provider, epochs = self.config.EPOCHS, validation_data = self.validation_data_provider, callbacks = [model_checkpoint_callback])
        
        return self.model

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        return

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
        return

    def predict(self, data):
        val_preds = self.model.predict(data)
        return val_preds
    
    def prediction_to_mask(self, val_pred):
        if (val_pred.shape == self.config.IMAGE_SIZE + (1,)):
            mask = val_pred
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
        else:
            mask = np.argmax(val_pred, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

        return mask.astype(np.uint8)

    