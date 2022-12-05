import json
from utils.rle_decoder import decode_rle

def parse_anotation(file_path):
    file = open(file_path)
    data = json.load(file)
    
    anotations = []
    for item in data:
        file_name = item['image']
        mask = decode_rle(item['tag'][0]['rle'], item['tag'][0]['original_height'], item['tag'][0]['original_width'])
        anotations.append((file_name, mask))
    
    return anotations