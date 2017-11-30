import numpy as np
import cv2
from matplotlib import pyplot as plt 
cv2.ocl.setUseOpenCL(False)

#img1_name = 'image_00000345_0_rect.png'
#img2_name = 'image_00000360_0_rect.png'
inp_c = 2#360
out_c = 1
img1_name = '1.png'
img2_name = '2.png'
    
#while img2_name != 'image_00001320_0_rect.png':
while inp_c <= 19:
    
    print img1_name, img2_name
    
    img1 = cv2.imread(img1_name,0)          # queryImage
    img2 = cv2.imread(img2_name,0)         # trainImage
    
    rows,cols = img1.shape
    
    # Initiate ORB detector
    
    orb = cv2.xfeatures2d.SURF_create()
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher()#cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors.
    matches = bf.knnMatch(des1,des2,k=2)
    #print type(matches)
    
    # Sort them in the order of their distance.
    #matches = sorted(matches, key = lambda x:x.distance)
    good1 = []
    good2 = []
    for m,n in matches:
        #print type(m)
        if m.distance < 0.75*n.distance:
            good1.append(m)
            good2.append([m])
            
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)
        
    
    # Draw first 10 matches.
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches ,None, flags=2)
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good2,None,flags=2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good2[0:3],None,flags=2)
    
    M = cv2.getAffineTransform(np.array([src_pts[0,0],src_pts[1,0],src_pts[2,0]]),np.array([dst_pts[0,0],dst_pts[1,0],dst_pts[2,0]]))
    #M = cv2.getPerspectiveTransform(np.array([src_pts[0,0],src_pts[1,0],src_pts[2,0],src_pts[3,0]]),np.array([dst_pts[0,0],dst_pts[1,0],dst_pts[2,0],dst_pts[3,0]]))
    
    print M
    
    
    
    dst = cv2.warpAffine(img1,M,(cols,rows))
    #dst = cv2.warpPerspective(img1,M,(cols,rows))
    
    #cv2.imwrite('o1.png',dst)
    #plt.imshow(img3),plt.show()
    #plt.imshow(img3)
    
    (thresh, img1_b) = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh, img2_b) = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh, dst_b) = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    frame_diff = cv2.absdiff(img2_b,img1_b)
    diff_2 = cv2.absdiff(img2_b,dst_b)
    
    #cv2.imshow('window1',dst)
    #cv2.imshow('original difference',frame_diff)
    #cv2.imshow('Compensated difference',diff_2)
    
    output_name = 'kp' + str(out_c) + '.png'
    output_name2 = 'p' + str(out_c) + '.png'
    cv2.imwrite(output_name,img3)
    cv2.imwrite(output_name2,frame_diff)
    out_c += 1
    inp_c += 1
    
    img1_name = img2_name
    img2_name = str(inp_c) + '.png'
#    if inp_c >= 1000:
#        img2_name = 'image' + '_0000' + str(inp_c) + '_0_rect.png'
#    else:
#        img2_name = 'image' + '_00000' + str(inp_c) + '_0_rect.png'
        
    
    #if cv2.waitKey(10000)==27:
        #break
    
#cv2.destroyAllWindows()