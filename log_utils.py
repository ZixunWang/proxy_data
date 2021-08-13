import os
import time
import shutil

from pathlib import Path
import numpy as np
import torch


class Logger(object):
    def __init__(self, cfg_file, log_dir, model_sample=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger_path = self.log_dir / 'log.txt'
        if not self.logger_path.exists():
            self.logger_file = open(self.logger_path, 'w')
        else:
            self.logger_file = open(self.logger_path, 'a')

        shutil.copy(cfg_file, self.log_dir / os.path.basename(cfg_file))

        self.data_sample_file = self.log_dir / 'data_sample.npy'
        self.state_file = self.log_dir / 'state.pth'
        self.model_sample_file = model_sample if model_sample is not None else self.log_dir / 'model_sample.npy'

    def load_file(self, name):
        if name == 'data sample':
            return np.load(str(self.data_sample_file))
        elif name == 'state':
            return torch.load(str(self.state_file))
        elif name == 'model sample':
            return np.load(str(self.model_sample_file))
        else:
            raise NotImplementedError

    def save_file(self, name, content):
        if name == 'data sample':
            np.save(self.data_sample_file, content)
        elif name == 'model sample':
            np.save(self.model_sample_file, content)
        elif name == 'state':
            torch.save(content, self.state_file)
        else:
            raise NotImplementedError

    def log(self, string, save=True):
        string = '{}: {}'.format(time.strftime('%d-%h-at-%H-%M-%S'), string)
        print(string)
        if save:
            self.logger_file.write(string+'\n')
            self.logger_file.flush()
