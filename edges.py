import numpy as np
import cv2

if __name__=="__main__":
    img_path = '/bigpen/Datasets/jhu-released/t4v3/100_depth.tiff'
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(img, threshold1=10, threshold2=100)
    _, markers = cv2.connectedComponents(edges)
    watershed = cv2.watershed(img, markers)

    wd, ht = edges.shape
    print(watershed.shape)
    watershed_color = cv2.cvtColor(watershed.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    watershed_color[:,:,0] = 0
    watershed_color[:,:,2] = 0
    # img_res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[watershed_color[:,:,1] > 0] = watershed_color
    img_res = cv2.resize(img, (int(ht/2), int(wd/2)))
    
    cv2.imshow('image', img_res)
    # cv2.imshow('watershed', watershed_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()