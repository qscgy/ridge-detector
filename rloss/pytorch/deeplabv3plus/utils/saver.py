import os
import shutil
import torch
from collections import OrderedDict
import glob
from natsort.natsort import natsorted
from random import choice

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = natsorted(glob.glob(os.path.join(self.directory, 'ex_*_*')))
        run_id = int(self.runs[-1].split('_')[-2]) + 1 if self.runs else 0

        # Assign names to experiments so they're easier to keep track of, a la Docker containers
        with open('names.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        name = choice(lines)

        self.experiment_dir = os.path.join(self.directory, 'ex_{}_{}'.format(str(run_id), name.lower()))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, '_parameters.txt')
        log_file = open(logfile, 'w')
        # p = OrderedDict()
        # p['datset'] = self.args.dataset
        # p['backbone'] = self.args.backbone
        # p['out_stride'] = self.args.out_stride
        # p['lr'] = self.args.lr
        # p['lr_scheduler'] = self.args.lr_scheduler
        # p['loss_type'] = self.args.loss_type
        # p['epoch'] = self.args.epochs
        # p['base_size'] = self.args.base_size
        # p['crop_size'] = self.args.crop_size
        # p['densecrfloss'] = self.args.densecrfloss
        # p['ncloss'] = self.args.ncloss
        # p['sigma_rgb_crf'] = self.args.sigma_rgb_crf
        p = vars(self.args)

        for key in p:
            log_file.write(key + ':' + str(p[key]) + '\n')
        log_file.close()