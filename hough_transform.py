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
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
grad = np.stack((sobelx, sobely), 2)
votes = hough_circle(grad, np.arange(1, 41, 10))

plt.imshow(votes)
plt.show()