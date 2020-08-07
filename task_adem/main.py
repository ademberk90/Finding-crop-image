import cv2
import numpy as np

def templateMatchingOpencvPreparedFunctions(img_temp,img_orig):
    print("templateMatchingOpencvPreparedFunctions")
    """ if you dont wanna use function you should read the image here """
    # img_orig = cv.imread("images/original.jpg")
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    """ if you dont wanna use function you should read the image here """
    # img_temp = cv.imread("images/temp_normal.jpg",0)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    """ take the information of width and height of images """
    w,h = img_temp.shape[::-1]

    """ Look for a any matching are on image """
    res = cv2.matchTemplate(img_gray, img_temp, cv2.TM_CCORR)

    """take the coordinates """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    top_right = (top_left[0] + w, top_left[1])
    bottom_left = (top_left[0], top_left[1] + h)
    bottom_right = (top_left[0] + w, top_left[1] + h)

    """ if you wanna see the are of matching images you should use these lines"""
    # cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
    # cv2.imshow('result',img_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("top left: {} \ntop right: {} \nbottom_left: {} \nbottom right: {}".format(top_left, top_right, bottom_left, bottom_right))
    print("\n --------------------------------------------------------------------\n")

def templateMatchingEqual(img_temp,img_orig):
    print("templateMatchingEqual")
    """ if you dont wanna use function you should read the image here """
    # img_temp = cv2.imread("flag.png")
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    """ if you dont wanna use function you should read the image here """
    # img_orig = cv2.imread("marioo.png")
    img_comp = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    """ take the information of width and height of images """
    width_temp, height_temp = img_temp.shape[::-1]
    width_orig, height_orig = img_comp.shape[::-1]
    top_left=None

    """ Look for a any matching are on image """
    print("Calculating")
    for w in range(width_orig - width_temp):
        for h in range(height_orig - height_temp):
            if np.array_equal(img_comp[h:h+height_temp,w:w+width_temp],img_temp):
                top_left = (w,h)
                break
    """ if any matching, take the coordinates """
    if top_left:
        bottom_right = (top_left[0] + width_temp, top_left[1] + height_temp)
        top_right = (top_left[0] + width_temp, top_left[1])
        bottom_left = (top_left[0], top_left[1] + height_temp)

        """ if you wanna see the are of matching images you should use these lines"""
        # cv2.rectangle(img_orig,top_left,bottom_right,(0,255,0),2)
        # cv2.imshow("result", img_orig)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("top left: {} \ntop right: {} \nbottom_left: {} \nbottom right: {}".format(top_left, top_right, bottom_left, bottom_right))
        print("\n --------------------------------------------------------------------\n")
    else:
        print("There is no matching")
        print("\n --------------------------------------------------------------------\n")


def templateMatchingMean(img_temp,img_orig):
    print("templateMatchingMean")
    """ if you dont wanna use function you should read the image here """
    # img_temp = cv2.imread("flag.png")
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    """ if you dont wanna use function you should read the image here """
    # img_orig = cv2.imread("marioo.png")
    img_comp = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    """ take the information of width and height of images """
    width_temp, height_temp = img_temp.shape[::-1]
    width_orig, height_orig = img_comp.shape[::-1]

    top_left= None
    dict_data = {}

    """ Look for a any matching are on image """
    print("Calculating")
    for w in range(width_orig - width_temp):
        for h in range(height_orig - height_temp):
            matr = img_comp[h:h + height_temp, w:w + width_temp] - img_temp
            np.absolute(matr)
            dict_data[(w, h)] = np.mean(matr)
    """ minimum mean or sum value give us the coordinate """
    min_val = min(dict_data.values())
    top_left = list(dict_data.keys())[list(dict_data.values()).index(min_val)]

    if top_left:
        bottom_right = (top_left[0] + width_temp, top_left[1] + height_temp)
        top_right = (top_left[0] + width_temp, top_left[1])
        bottom_left = (top_left[0], top_left[1] + height_temp)
        # cv2.rectangle(img_orig, top_left, bottom_right, (0, 255, 0), 2)
        # cv2.imshow("result", img_orig)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("top left: {} \ntop right: {} \nbottom_left: {} \nbottom right: {}".format(top_left, top_right, bottom_left, bottom_right))
        print("\n --------------------------------------------------------------------\n")
    else:
        print("There is no matching")
        print("\n --------------------------------------------------------------------\n")


def featureMatchingSirf(img_temp, img_orig):
    print("featureMatchingSirf")
    """ set a condition that at least 10 matches are to be there to find the object. """
    MIN_MATCH_COUNT = 5

    """ if you dont wanna use function you should read the image here """
    # img_orig = cv.imread("images/original.jpg")
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    """ if you dont wanna use function you should read the image here """
    # img_temp = cv.imread("images/temp_normal.jpg",0)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    """ initiate SIFT detector """
    sift = cv2.xfeatures2d.SIFT_create()

    """ find the keypoints and descriptors with SIFT """
    kp1, des1 = sift.detectAndCompute(img_temp, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    """ BFMatcher with default params """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    """Apply ratio test"""
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    """If enough matches are found, we extract the locations of matched keypoints in both the images"""
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img_temp.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        """ if you wanna see the are of matching images you should use these lines"""
        # cv2.polylines(img_orig, [np.int32(dst)], True, 255, 2, cv2.LINE_AA)
        # cv2.imshow("result", img_orig)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        top_left, top_right, bottom_left, bottom_right = np.int32(dst)

        top_left = top_left[0][0], top_left[0][1]
        top_right = top_right[0][0], top_right[0][1]
        bottom_left = bottom_left[0][0], bottom_left[0][1]
        bottom_right = bottom_right[0][0], bottom_right[0][1]

        print("top left: {} \ntop right: {} \nbottom_left: {} \nbottom right: {}".format(top_left, top_right, bottom_left, bottom_right))
        print("\n --------------------------------------------------------------------\n")

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        print("\n --------------------------------------------------------------------\n")

img_temp = cv2.imread("images/Small_area.png")
img_orig = cv2.imread("images/StarMap.png")
#Func1
templateMatchingOpencvPreparedFunctions(img_temp,img_orig)
#Func2
templateMatchingEqual(img_temp, img_orig)
#Func3
templateMatchingMean(img_temp,img_orig)
#Func4
featureMatchingSirf(img_temp,img_orig)