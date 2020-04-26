__author__ = 'Brian M Anderson'

# Created on 4/15/2020
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


class Add_Images_and_LR(Callback):
    def __init__(self, log_dir, add_images=True, validation_data=None, number_of_images=3):
        super(Add_Images_and_LR, self).__init__()
        assert add_images and validation_data is not None, 'Need to provide validation data if you want images!'
        self.add_images = add_images
        self.validation_data = iter(validation_data)
        self.number_of_images = number_of_images
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val_images'))

    def return_proper_size(self, val):
        if tf.shape(val)[0] != 1:
            val = tf.expand_dims(val, axis=0)
        if tf.shape(val)[-1] != 1:
            val = tf.expand_dims(val, axis=-1)
        return val

    def scale_0_1(self, val):
        val = tf.subtract(val, tf.reduce_min(val))
        val = tf.divide(val, tf.reduce_max(val))
        return val

    def write_images(self, epoch):
        output_x = []
        output_y = []
        output_pred = []
        for i in range(self.number_of_images):
            x, y = next(self.validation_data)
            y = tf.squeeze(y)
            indexes = tf.unique(tf.where(y > 0)[..., 0])[0]
            index = indexes[tf.shape(indexes)[0] // 2]
            x = tf.expand_dims(x[index,...],axis=0)
            y = y[index, ...]
            pred = self.model(x, training=False)
            pred = tf.argmax(pred, axis=-1)
            pred_out = pred[0,...]
            x = x[...,0]
            x, y, pred_out = self.return_proper_size(x), self.return_proper_size(y), self.return_proper_size(pred_out)
            x, y, pred_out = tf.cast(x, 'float32'), tf.cast(y, 'float32'), tf.cast(pred_out, 'float32')
            x, y, pred_out = self.scale_0_1(x), self.scale_0_1(y), self.scale_0_1(pred_out)
            output_x.append(x)
            output_y.append(y)
            output_pred.append(pred_out)
        x, y, pred_out = tf.concat(output_x, axis=2), tf.concat(output_y, axis=2), tf.concat(output_pred, axis=2)
        return x, y, pred_out

    def on_epoch_end(self, epoch, logs=None):
        if self.add_images:
            x, y, pred_out = self.write_images(epoch=epoch)
            with self.file_writer.as_default():
                tf.summary.image('Image', tf.cast(x, 'float32'), step=epoch)
                tf.summary.image('Truth', tf.cast(y, 'float32'), step=epoch)
                tf.summary.image('Pred', tf.cast(pred_out, 'float32'), step=epoch)
                tf.summary.scalar('Learning_Rate',tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)
        else:
            with self.file_writer.as_default():
                tf.summary.scalar('Learning_Rate',tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)


if __name__ == '__main__':
    pass
