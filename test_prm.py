import torch
from PRM.datasets.VOCClassification import pascal_voc_object_categories
from PRM.prm import fc_resnet50, peak_response_mapping
from PRM.modules.peak_response_mapping import imresize
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import PIL
import json
import matplotlib.pyplot as plt

if __name__=='__main__':
    class_names = pascal_voc_object_categories()
    print('Object categories: ' + ', '.join(class_names))

    image_size = 448
    # image pre-processor
    transformer = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(transformer)

    backbone = fc_resnet50(20, False)
    model = peak_response_mapping(backbone)

    state = torch.load('latest_prm.pt')
    model = state
    model = model.cuda()

    idx = 0
    raw_img = PIL.Image.open('./PRM/demo/data/sample%d.jpg' % idx).convert('RGB')
    input_var = transformer(raw_img).unsqueeze(0).cuda().requires_grad_()
    
    model = model.eval()
    print('Object categories in the image:')
    confidence = model(input_var)

    for idx in range(len(class_names)):
        if confidence.data[0, idx] > 0:
            print('    [class_idx: %d] %s (%.2f)' % (idx, class_names[idx], confidence[0, idx]))

    model = model.inference()
    visual_cues = model(input_var)
    if visual_cues is None:
        print('No class peak response detected')
    else:
        confidence, class_response_maps, class_peak_responses, peak_response_maps = visual_cues
        _, class_idx = torch.max(confidence, dim=1)
        class_idx = class_idx.item()
        num_plots = 2 + len(peak_response_maps)
        f, axarr = plt.subplots(1, num_plots, figsize=(num_plots * 4, 4))
        axarr[0].imshow(raw_img.resize((image_size, image_size)))
        axarr[0].set_title('Image')
        axarr[0].axis('off')
        axarr[1].imshow(class_response_maps[0, class_idx].cpu(), interpolation='bicubic')
        axarr[1].set_title('Class Response Map ("%s")' % class_names[class_idx])
        axarr[1].axis('off')
        for idx, (prm, peak) in enumerate(sorted(zip(peak_response_maps, class_peak_responses), key=lambda v: v[-1][-1])):
            axarr[idx + 2].imshow(prm.cpu(), cmap=plt.cm.jet)
            axarr[idx + 2].set_title('Peak Response Map ("%s")' % (class_names[peak[1].item()]))
            axarr[idx + 2].axis('off')
    plt.show()