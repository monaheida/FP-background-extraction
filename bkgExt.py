'''
Using region growing to subtract the background and show the border of the image.

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

    def region_growing(self, image, seed_point, threshold=100):
        region = np.zeros_like(image, dtype=np.uint8)
        region_points = [seed_point]
        while region_points:
            y, x = region_points.pop()
            if region[y, x] == 0:
                region[y, x] = 255
                neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                for ny, nx in neighbors:
                    if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                        if abs(int(image[ny, nx]) - int(image[y, x])) < threshold:
                            region_points.append((ny, nx))
        return region

    def subtractBackground(self, image):
        print("Extracting background...")
        preprocessed = self.backgroundPreprocessing(image)
        outImage = image.copy()

        # Use region growing to determine the fingerprint area
        seed_point = (0, 0)  # You can choose a seed point inside the fingerprint area
        region = self.region_growing(preprocessed, seed_point)

        # Update the output image with the background subtracted
        outImage[region == 255] = 255

        return region, outImage

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

