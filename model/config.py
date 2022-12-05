
class Config(object):
    BATCH_SIZE = 32

    EPOCHS = 150

    IMAGE_SIZE = (528, 297)

    NUM_CLASSES = 2 # (0 = Backgroun, 1 = Building)

    IMAGE_FOLDER_PATH = 'data/images'

    AUGMENTATION_ENABLE = False

