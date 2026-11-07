from face_landmark_detection import generate_face_correspondences
from delaunay_triangulation import make_delaunay
from face_morph import generate_morph_sequence

import cv2

def doMorphing(img1, img2, detector, predictor, frames=100):
    	
	image1 = cv2.imread(img1)
	image2 = cv2.imread(img2)

	[size, img1, img2, points1, points2, list3] = generate_face_correspondences(image1, image2, detector, predictor)

	tri = make_delaunay(size[1], size[0], list3)

	generate_morph_sequence(frames, img1, img2, points1, points2, tri)

