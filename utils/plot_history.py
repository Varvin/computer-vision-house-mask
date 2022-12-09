from matplotlib import pyplot as plt

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))
    
    if 'val_loss' in h.keys():
        plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
        plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
        plt.legend(['Train', 'Validation'])
        plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-',
                                epochs, h['val_accuracy'], '.-')
        plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
            
        print('Train Acc     ', h['accuracy'][-1])
        print('Validation Acc', h['val_accuracy'][-1])
    else:
        plt.subplot(121), plt.plot(epochs, h['loss'], '.-')
        plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
        plt.legend(['Train', 'Validation'])
        plt.subplot(122), plt.plot(epochs, h['accuracy'], '.-')
        plt.grid(True), plt.xlabel('epochs'), plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
            
        print('Train Acc     ', h['accuracy'][-1])