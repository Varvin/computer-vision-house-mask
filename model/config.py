
class Config(object):
    BATCH_SIZE = 15

    EPOCHS = 150

    IMAGE_SIZE = (256, 256) #(528, 297)

    NUM_CLASSES = 2 # (0 = Backgroun, 1 = Building)

    IMAGE_FOLDER_PATH = 'data/images'

    AUGMENTATION_ENABLE = False

