# Please note reference to call back in CyclicLR_onecycle, this has been slightly adapted from the original model
import numpy as np
from tensorflow.python.keras.callbacks import Callback
import warnings
from tensorflow.python.keras.backend import get_value
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = np.sum(y_true[...,1:] * y_pred[...,1:])
    union = np.sum(y_true[...,1:]) + np.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)

class ModelCheckpoint_new(Callback):

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, model=None, save_best_and_all=False):
        self.is_gpu_model = False
        if model:
            self.save_model = model
            self.is_gpu_model = True
        super(ModelCheckpoint_new, self).__init__()
        self.save_best_and_all = save_best_and_all
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def set_path(self, path):
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if not self.is_gpu_model:
            self.save_model = self.model
        self.epoch = epoch + 1
        self.epochs_since_last_save += 1
        filepath = self.filepath.replace('{epoch:02d}', str(epoch + 1))
        if self.save_best_only or self.save_best_and_all:
            filepath_best = filepath.replace('.hdf5', '_best.hdf5')
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, self.monitor, self.best,
                                 current, filepath_best))
                    self.best = current
                    if self.save_weights_only:
                        self.model.save_weights(filepath_best, overwrite=True)
                    else:
                        self.model.save(filepath_best, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))
        if self.epochs_since_last_save >= self.period:
            if not self.save_best_only or self.save_best_and_all:
                self.epochs_since_last_save = 0
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.save_model.save_weights(filepath, overwrite=True)
                else:
                    self.save_model.save(filepath, overwrite=True)


class Add_LR_To_Tensorboard(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs is not None:
            logs['learning_rate'] = get_value(self.model.optimizer.lr)
        return logs