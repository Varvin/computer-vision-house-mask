import numpy as np

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)

def access_bit(data, num):
    """ from bytes array to bits by num position
    """
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data):
    """ get bit string from bytes data
    """
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])

def decode_rle(rle, original_height, original_width, print_params: bool = False):
    """ from LS RLE to numpy uint8 3d image [width, height, channel]
    
    Args:
        print_params (bool, optional): If true, a RLE parameters print statement is suppressed
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]
    
    if print_params:
        print('RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes')
        
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    out = np.reshape(out, [original_height, original_width, 4])[:,:,3]
    out[out>0] = 1
    out = np.reshape(out,[original_height, original_width, 1])
    return out