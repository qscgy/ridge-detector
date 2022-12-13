import numpy as np
import cv2
import matplotlib.pyplot as plt

def hough_circle(grad, rs):
    ii = np.arange(grad.shape[0])
    xs, ys = np.meshgrid(np.arange(grad.shape[1]), np.arange(grad.shape[0]), indexing='xy')
    coords = np.stack((xs, ys), 2)
    grad_norm = np.sqrt(np.sum(grad**2))
    grad_dir = grad/grad_norm
    rs = rs.reshape(-1,1,1,1)

    vote_coords = (coords-grad_dir)*rs
    vote_coords = np.round(vote_coords).astype(int)
    vote_coords = np.reshape(vote_coords, (len(rs),-1,2))
    print(vote_coords.shape)
    votes = np.zeros((len(rs), *xs.shape))
    for i in range(len(rs)):
        votes[i][vote_coords[i,:,0]][vote_coords[i,:,1]] = np.tan(grad_norm.flatten()-0.1)
    return votes

img = cv2.imread('labels.png')
src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)
# grad = np.stack((sobelx, sobely), 2)
# laplacian = cv2.Laplacian(src, cv2.CV_16S, ksize=5)

# polar = cv2.linearPolar(img, (130, 139), np.sqrt(img.shape[0]**2 + img.shape[1]**2), cv2.WARP_FILL_OUTLIERS)
# squashed = cv2.resize(src, (src.shape[0], src.shape[1]//4))

circles = cv2.HoughCircles(src,cv2.HOUGH_GRADIENT,1.5,20,
                            param1=120,param2=50,minRadius=60)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)


cv2.imshow('Original', img)
# cv2.imshow('Squashed', squashed)
cv2.waitKey(0)

# votes = hough_circle(grad, np.arange(1, 41, 10))

# plt.imshow(votes)
# plt.show()