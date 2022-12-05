import tensorflow as tf
from abc import ABC, abstractmethod

class AbstractModelProvider(ABC):

    @abstractmethod
    def get_model(self, img_size, num_classes) -> tf.keras.Model:
        pass
