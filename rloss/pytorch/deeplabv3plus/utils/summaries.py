import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from ..dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer
    
    def visualize_prob(self, writer, dataset, probs, global_step):
        import numpy as np
        from matplotlib import cm
        x = probs[:3,1].unsqueeze(1).cpu().numpy()
        cmap = np.apply_along_axis(cm.jet, 0, x)
        cmap = torch.from_numpy(cmap)
        img = make_grid(cmap.data, 3, normalize=False)
        writer.add_image('P(is fold region)', img, global_step)

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        input_img = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', input_img, global_step)
        pred_map = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', pred_map, global_step)
        gt_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', gt_image, global_step)

        all_img = torch.cat((input_img, pred_map, gt_image), -2)
        writer.add_image('All three', all_img, global_step)
        
        # input_img[:,0] *= (1-torch.sign(pred_map[:,0]))
        # writer.add_image('Overlaid label', input_img, global_step)