import cv2
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
from utils.augmentation import flip, blur, rotate
from model.config import Config

class DataProvider(keras.utils.Sequence):

    config : Config

    def __init__(self, config : Config, annotations):
        self.config = config
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations) // self.config.BATCH_SIZE

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to **batch** index."""
        i = idx * self.config.BATCH_SIZE
        batch_annotations = self.annotations[i : i + self.config.BATCH_SIZE]
        
        x = np.zeros((self.config.BATCH_SIZE,) + self.config.IMAGE_SIZE + (3,), dtype="float32")
        y = np.zeros((self.config.BATCH_SIZE,) + self.config.IMAGE_SIZE + (1,), dtype="uint8")
        for j, data in enumerate(batch_annotations):
            file_name = data[0]
            mask = data[1]
            image = cv2.cvtColor(cv2.imread(f'{self.config.IMAGE_FOLDER_PATH}/{file_name}'), cv2.COLOR_BGR2RGB)
            image = image/255

            if(self.config.AUGMENTATION_ENABLE and np.random.rand() > 0.5):
                type = np.random(3)
                if(type == 0):
                    image, mask = rotate(image, mask)
                elif(type == 1):
                    image = blur(image)
                else:
                    image, mask = flip(image, mask)
            
            x[j] = image
            y[j] = mask
            
        return x, y
    
    def on_epoch_end(self):
        self.annotations = shuffle(self.annotations)
