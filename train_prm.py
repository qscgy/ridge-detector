from tokenize import group
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, Normalize
from PRM.datasets import pascal_voc_classification
from torch.utils.data import DataLoader, random_split
from torch.optim.sgd import SGD
import torch
from PRM.models.fc_resnet import FC_ResNet
from PRM.prm import peak_response_mapping
import torch.nn.functional as F
from fnmatch import fnmatch
from torch.utils.tensorboard import SummaryWriter

voc_root = '/bigpen/VOC2012/VOCdevkit/'
voc_year = '2012'
im_size = [448, 448]
im_mean = [0.485, 0.456, 0.406]
im_std = [0.229, 0.224, 0.225]
batch_size = 16
n_epoch = 20

# From Zhou et al. PRM
def multilabel_soft_margin_loss(
    input, 
    target,
    weight = None,
    size_average = True,
    reduce = True,
    difficult_samples = True):
    """Multilabel soft margin loss.
    """

    if difficult_samples:
        # label 1: positive samples
        # label 0: difficult samples
        # label -1: negative samples
        gt_label = target.clone()
        gt_label[gt_label == 0] = 1
        gt_label[gt_label == -1] = 0
    else:
        gt_label = target
        
    return F.multilabel_soft_margin_loss(input, gt_label, weight, size_average, reduce)

def finetune(model, base_lr, groups, ignore_the_rest=False, raw_query=False):
    """Fintune.
    """

    parameters = [dict(params=[], names=[], query=query if raw_query else '*'+query+'*', lr=lr*base_lr) for query, lr in groups.items()]
    rest_parameters = dict(params=[], names=[], lr=base_lr)
    for k, v in model.named_parameters():
        for group in parameters:
            if fnmatch(k, group['query']):
                group['params'].append(v)
                group['names'].append(k)
            else:
                rest_parameters['params'].append(v)
                rest_parameters['names'].append(k)
    if not ignore_the_rest:
        parameters.append(rest_parameters)
    for group in parameters:
        group['params'] = iter(group['params'])
    return parameters

def train(epoch, model, train_loader, optimizer, criterion, writer=None, global_step=0):
    running_loss = 0.0
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += len(inputs)

        if i % 20 == 19:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.5f}')
            if writer is not None:
                writer.add_scalar('loss', running_loss/20, global_step)
            running_loss = 0.0
    return global_step

def validate(epoch, model, val_loader, criterion, writer=None, global_step=0):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, targets in val_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        print(f'[{epoch + 1}] validation loss: {running_loss / len(val_loader):.5f}')
        
        if writer is not None:
            # This used to be logging running_loss/20, which is why it looks so high
            writer.add_scalar('loss', running_loss/len(val_loader), global_step)

def main():
    train_transform = Compose([
        Resize(im_size),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(im_mean, im_std),
    ])

    trainval = pascal_voc_classification(
        'trainval',
        voc_root,
        year=2012,
        transform=train_transform,
    )
    n_samples = len(trainval)
    n_train = int(n_samples * 0.9)
    
    train_dataset, val_dataset = random_split(trainval, [n_train, n_samples-n_train], generator=torch.Generator().manual_seed(12))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    backbone = FC_ResNet(resnet50(ResNet50_Weights.DEFAULT), 20)
    backbone = backbone.cuda()

    prm = peak_response_mapping(backbone,
                                win_size=3,
                                sub_pixel_locating_factor=8)
    prm = prm.cuda()
    params = finetune(prm, base_lr=0.01, groups={'features': 0.01})
    optimizer = SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = multilabel_soft_margin_loss

    train_writer = SummaryWriter('results/debug/train')
    val_writer = SummaryWriter('results/debug/val')
    global_step = 0

    for epoch in range(n_epoch):
        global_step = train(epoch, prm, train_loader, optimizer, criterion, writer=train_writer, global_step=global_step)
        validate(epoch, prm, val_loader, criterion, writer=val_writer, global_step=global_step)
        torch.save(prm, 'latest_prm.pt')


if __name__=='__main__':
    main()