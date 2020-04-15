__author__ = 'Brian M Anderson'
# Created on 4/15/2020
import tensorflow as tf
import os


class Add_Images(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, number_of_images=3):
        super(Add_Images, self).__init__()
        self.validation_data = iter(validation_data)
        self.number_of_images = number_of_images
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir,'val_images'))

    def return_proper_size(self, val):
        if tf.shape(val)[0] != 1:
            val = tf.expand_dims(val, axis=0)
        if tf.shape(val)[-1] != 1:
            val = tf.expand_dims(val, axis=-1)
        return val

    def on_epoch_end(self, epoch, logs=None):
        output_x = []
        output_y = []
        output_pred = []
        for i in range(self.number_of_images):
            x, y = next(self.validation_data)
            y = tf.squeeze(y)
            pred = self.model.predict(x)
            pred = tf.argmax(pred,axis=-1)
            indexes = tf.unique(tf.where(y>0)[...,0])[0]
            index = indexes[tf.shape(indexes)[0]//2]
            pred_out = pred[index,...]
            x = x[index,...,0]
            y = y[index,...]
            x, y, pred_out = self.return_proper_size(x), self.return_proper_size(y), self.return_proper_size(pred_out)
            output_x.append(x)
            output_y.append(y)
            output_pred.append(pred_out)
        x, y, pred_out = tf.concat(output_x,axis=2), tf.concat(output_y, axis=2), tf.concat(output_pred, axis=2)
        with self.file_writer.as_default():
            tf.summary.image('Image', tf.cast(x, 'float32'), step=epoch)
            tf.summary.image('Truth', tf.cast(y,'float32'), step=epoch)
            tf.summary.image('Pred', tf.cast(pred_out,'float32'), step=epoch)


if __name__ == '__main__':
    pass
