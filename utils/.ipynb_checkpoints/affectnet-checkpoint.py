import os
import csv
from PIL import Image

class AFItem:
    def __init__(self, attributes):
        self.path = attributes[0]
        self.face_x = attributes[1]
        self.face_y = attributes[2]
        self.width = attributes[3]
        self.height = attributes[4]
        self.landmark = attributes[5]
        self.expression = attributes[6]
        self.valence = attributes[7]
        self.arousal = attributes[8]

    def self_crop(self, work_dir):
        path = os.path.join(work_dir, self.path)
        img = Image.open(path)
        ROI = [self.face_x, self.face_y,
               self.face_x+self.width,
               self.face_y+self.height]
        cropped = img.crop(ROI)
        return cropped



if __name__ == '__main__':
    pass