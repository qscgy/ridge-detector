from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from data import FoldSegmentation
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 216
    args.crop_size = 216
    args.base_dir = '/playpen/Datasets/scribble-samples/'

    voc_train = FoldSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=1, shuffle=True, num_workers=0)
    print(len(dataloader))
    for ii, sample in enumerate(dataloader):
        img = sample['image'].numpy()
        gt = sample['label'].numpy()
        for jj in range(sample["image"].size()[0]):
            tmp = np.array(gt[jj]).astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            img_tmp[tmp==0] = np.array([255,0,0])
            img_tmp[tmp==1] = np.array([0,255,0])       

            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(tmp)

        if ii == 1:
            break

    plt.show(block=True)