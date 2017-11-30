import numpy as np
import cv2
from matplotlib import pyplot as plt 
from sys import argv
cv2.ocl.setUseOpenCL(False)

# params
USE_PERSPECTIVE_TRANSFORM = '-p' in argv

#img1_name = 'image_00000345_0_rect.png'
#img2_name = 'image_00000360_0_rect.png'
inp_c = 2#360
out_c = 1
inp_cMax = 7#1320
img1_name = '1.png'
img2_name = '2.png'

while inp_c <= inp_cMax:
    img1 = cv2.imread(img1_name,0)          # queryImage
    img2 = cv2.imread(img2_name,0)         # trainImage
    
    rows,cols = img1.shape
    
    
    orb = cv2.xfeatures2d.SURF_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher()#cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    
    
    # Sort them in the order of their distance.
    best_thresh = 0
    best_compensation = None
    best_diff = np.inf
    for distance_threshold in np.linspace(0., 1., 101):
    	good1 = []
    	for m, n in matches:
    	    if m.distance < distance_threshold * n.distance:
    	        good1.append(m)
    	        
    	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1, 1, 2)
    	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good1 ]).reshape(-1, 1, 2)
    	    
    
    	# Draw first 10 matches.
    	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches ,None, flags=2)
    	matches_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, map(lambda x: [x], good1), None, flags=2)
    
    	try:
    		if USE_PERSPECTIVE_TRANSFORM:
    			M = cv2.getPerspectiveTransform(np.array([src_pts[0,0], src_pts[1,0], src_pts[2,0], src_pts[3,0]]),
    											np.array([dst_pts[0,0], dst_pts[1,0], dst_pts[2,0], dst_pts[3,0]]))
    
    			transformed_image = cv2.warpPerspective(img1, M, (cols, rows))
    		else:
    			M = cv2.getAffineTransform(np.array([src_pts[0,0], src_pts[1,0], src_pts[2,0]]),
    									   np.array([dst_pts[0,0], dst_pts[1,0], dst_pts[2,0]]))
    
    			transformed_image = cv2.warpAffine(img1, M, (cols, rows))
    	except(IndexError): # if there are no matches
    		continue
    
    	# cv2.imwrite('o1.png',transformed_image)
    
    	# compute differences
    	compensated_diff = cv2.absdiff(img2, transformed_image)
    	difference_norm = np.linalg.norm(compensated_diff)
    	
    	if difference_norm < best_diff:# assuming we want to minimize difference between transformed im1 and im2
    		best_thresh = distance_threshold
    		best_transform = transformed_image
    		best_diff = difference_norm
    
    
    if USE_PERSPECTIVE_TRANSFORM: print 'Using PERSPECTIVE transform we obtain:'
    else: print 'Using AFFINE transform we obtain:'
    
    compensated_diff = cv2.absdiff(img2, best_transform)
    original_diff = cv2.absdiff(img2, img1)
    print 'Best distance thresh = %0.2f\nBest difference norm = %0.4f\nOriginal difference norm = %0.4f\n'%(best_thresh, best_diff, np.linalg.norm(original_diff)) 

    (thresh, out_img) = cv2.threshold(compensated_diff, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    output_name = 'o' + str(out_c) + '.png'
    cv2.imwrite(output_name,out_img)
    out_c += 1
    inp_c += 1#15
    
    img1_name = img2_name
    img2_name = str(inp_c) + '.png'
    
#    if inp_c >= 1000:
#        img2_name = 'image' + '_0000' + str(inp_c) + '_0_rect.png'
#    else:
#        img2_name = 'image' + '_00000' + str(inp_c) + '_0_rect.png'
        
    
#fig = plt.figure()
#fig.add_subplot(1,3,1)
#plt.imshow(best_transform)
#plt.title('Compensated second image')

#fig.add_subplot(1,3,2)
#plt.imshow(compensated_diff)
#plt.title('Compensated frame difference')
#
#fig.add_subplot(1,3,3)
#plt.imshow(original_diff)
#plt.title('Original frame difference')
#
#plt.show()

#cv2.imshow('compendated_diff',compensated_diff)
#cv2.imshow('original_diff',original_diff)
#
#if cv2.waitKey(1000)==27:
#	cv2.destroyAllWindows()