'''
Using the CCL algorithms to subtract the background and show the border of the image.
CCL is faster than region growing algorithm.

'''

import cv2
import numpy as np
from imgPrc import BasicImgOperations

class BckgrExtractor:
    
    def __init__(self):
        self.basic = BasicImgOperations()

    def backgroundPreprocessing(self, image):
        print("Background preprocessing...")
        result = self.basic.threshold(image, thresh=210)
        result = self.basic.open(result, elemSize=7)
        result = self.basic.erode(result, elemSize=15)
        result = self.basic.open(result, elemSize=7)
        result = self.basic.dilate(result, elemSize=17)
        result = self.basic.close(result, elemSize=5)
        result = cv2.GaussianBlur(result, (13, 13), 0, 0)
        result = self.basic.threshold(result, thresh=100)
        return result

    def subtractBackground(self, image):
        print("Extracting background...")
        preprocessed = self.backgroundPreprocessing(image)
        outImage = image.copy()

        # Use Connected Component Labeling to determine the damaged areas
        num_labels, labeled_img = cv2.connectedComponents(preprocessed)

        # Filter out the background (largest) component, which corresponds to the fingerprint area
        max_area = 0
        max_label = 0
        for label in range(1, num_labels):  # Start from 1 as label 0 corresponds to the background
            area = np.sum(labeled_img == label)
            if area > max_area:
                max_area = area
                max_label = label

        # Create the background mask
        background_mask = (labeled_img == max_label).astype(np.uint8) * 255

        # Update the output image with the background subtracted
        outImage[background_mask == 255] = 255

        return background_mask, outImage

    def findBorders(self, image):
        print("Finding image borders...")
        preprocessed = self.basic.threshold(image, thresh=230)
        preprocessed = cv2.erode(preprocessed, np.ones((9, 9), dtype=np.uint8))
        preprocessed = self.basic.open(preprocessed, elemSize=5)
        preprocessed = cv2.dilate(preprocessed, np.ones((11, 11), dtype=np.uint8))
        preprocessed = self.basic.blur(preprocessed)
        preprocessed = cv2.threshold(preprocessed, 100, 255, cv2.THRESH_BINARY_INV)[1]

        extended = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
        extended_pre = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
        extended[1:-1, 1:-1] = image
        extended_pre[1:-1, 1:-1] = preprocessed

        contours, _ = cv2.findContours(extended_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        result = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        cv2.drawContours(result, contours, -1, (0), 6)
        return result

if __name__ == '__main__':
    image = cv2.imread('/path-to-img', cv2.IMREAD_GRAYSCALE)

    bckgr_extractor = BckgrExtractor()

    background_map, background_subtracted_img = bckgr_extractor.subtractBackground(image)
    border_img = bckgr_extractor.findBorders(image)
    
    cv2.imshow('Background Subtracted Image', background_subtracted_img)
    cv2.imshow('Border Image', border_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

