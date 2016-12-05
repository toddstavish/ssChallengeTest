#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from collections import namedtuple
import csv
import os.path
import random
import sys

import numpy as np
import PIL.Image
import shapely.geometry
import shapely.wkt
import skimage.segmentation


def detect(model, image_path):
    '''
    Find few potential building candidates within an image, and return a list of tuple
    (polygon_wkt, confidence) of detections
    '''
    detections = []
    image = np.array(PIL.Image.open(image_path))
    segments = skimage.segmentation.felzenszwalb(image, scale=200.0, sigma=0.75, min_size=1)
    counts = np.bincount(segments.flatten())
    for i, size in enumerate(counts):
        if model.max_size >= size and size >= model.min_size:
            points = shapely.geometry.MultiPoint([(y, x, 0) for x,y in np.argwhere(segments == i)])
            hull = points.convex_hull
            # simple area and confidence heuristic based on polygon area and length
            if (model.hratio0 * len(points) >= hull.area and
                hull.area > model.hratio1 * hull.length):
                # confidence formula
                confidence = int(1000.0 * hull.area / hull.length)
                # use shapely.wkt to build the polygon string
                detections.append((shapely.wkt.dumps(hull, old_3d = True, rounding_precision = 2), confidence))
    # limit to max_detections per image based on confidence scores
    if len(detections) > model.max_detections:
        detections.sort(key=lambda x: x[1], reverse = True)
        detections = detections[:model.max_detections]
    return detections


def main(args):
    '''
    For all images, detect buildings and write results to a CSV file
    '''
    # image_folder = '../spacenet_TrainData/3band'  # folder with test images
    image_folder = '../spacenet_TestData/3band'  # folder with test images
    image_prefix_length = 6 # prefix to be removed is "3band_"
    image_subset = 100      # number of images to process (or 0 for all images)

    # load offline trained parameters (could also be a disk file with millions of parameters)
    Model = namedtuple('Model',
        ['max_size', 'min_size', 'hratio0', 'hratio1', 'max_detections'])
    model = Model(max_size = 2000, min_size = 20, hratio0 = 1.25,
        hratio1 = 3.0, max_detections = 75)

    image_list = [x for x in os.listdir(image_folder) if x[-4:]=='.tif']
    if image_subset and image_subset < len(image_list):
        random.shuffle(image_list)
        image_list = image_list[:image_subset]
    # create a csv file, which will comply with the contest format
    with open('submission.csv', 'w') as dest:
        writer = csv.writer(dest)
        # header line
        writer.writerow(['ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])
        # loop over test images and detect building based on model
        for j, basename in enumerate(image_list):
            image_name = basename[image_prefix_length:-4]
            detections = detect(model, os.path.join(image_folder, basename))
            # write to submission file
            if detections :
                for i, detection in enumerate(detections):
                    polygon_wkt = detection[0]
                    confidence = detection[1]
                    writer.writerow([image_name, i, polygon_wkt, confidence])
            else :
                writer.writerow([image_name, -1, "POLYGON EMPTY", -1])
            # summary data for the image
            print("{}, {}, {} polygons".format(j, image_name, len(detections)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
