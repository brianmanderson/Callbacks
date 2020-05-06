__author__ = 'Brian M Anderson'
# Created on 4/15/2020
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix


class Add_Images_and_LR(Callback):
    def __init__(self, log_dir, add_images=True, validation_data=None, number_of_images=3):
        super(Add_Images_and_LR, self).__init__()
        if add_images and validation_data is None:
            AssertionError('Need to provide validation data if you want images!')
        self.add_images = add_images
        self.number_of_images = number_of_images
        self.validation_data = iter(validation_data)
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val_images'))
        # if add_images:
        #     self.create_image_set(iter(validation_data))

    def create_image_set(self, validation_data):
        self.image_dict = {}
        for i in range(self.number_of_images):
            print('Preparing out image {}'.format(i))
            x, y = next(validation_data)
            self.image_dict[i] = [x, y]

    def return_proper_size(self, val):
        val = tf.squeeze(val)
        if len(val.shape) > 2:
            val = val[0]
        if tf.shape(val)[0] != 1:
            val = tf.expand_dims(val, axis=0)
        if tf.shape(val)[-1] != 1:
            val = tf.expand_dims(val, axis=-1)
        return val

    def scale_0_1(self, val):
        val = tf.subtract(val, tf.reduce_min(val))
        val = tf.divide(val, tf.reduce_max(val))
        return val

    def write_images(self):
        output_x = []
        output_y = []
        output_pred = []
        print('Writing out images')
        for i in range(self.number_of_images):
            x, y_base = next(self.validation_data)
            print('Writing out image {}'.format(i))
            # x, y_base = self.image_dict[i]
            y = y_base
            start_index = None
            while type(y) is tuple:
                y = y[0]
            y = tf.squeeze(y)
            if len(y.shape) > 2 and y.shape[0] > 32:
                indexes = tf.unique(tf.where(y > 0)[..., 0])[0]
                start_index = indexes[tf.shape(indexes)[0] // 2]
            if start_index is not None:
                x_pred = []
                if type(x) is tuple:
                    for i in x:
                        expand_0 = i.shape[0] == 1
                        end_shape = i.shape[-1]
                        i = tf.squeeze(i)
                        i = i[tf.maximum(start_index-16,0):tf.minimum(start_index+16,i.shape[0])]
                        if expand_0:
                            i = tf.expand_dims(i,axis=0)
                        if end_shape == 1:
                            i = tf.expand_dims(i,axis=-1)
                        x_pred.append(i)
                    x = tuple(x_pred)
                y_pred = []
                if type(y_base) is tuple:
                    for i in y_base:
                        expand_0 = i.shape[0] == 1
                        end_shape = i.shape[-1]
                        i = tf.squeeze(i)
                        i = i[tf.maximum(start_index-16,0):tf.minimum(start_index+16,i.shape[0])]
                        if expand_0:
                            i = tf.expand_dims(i,axis=0)
                        if end_shape == 1:
                            i = tf.expand_dims(i,axis=-1)
                        y_pred.append(i)
                    y_base = tuple(y_pred)
            pred_base = self.model(x, training=False)
            while type(x) is tuple:
                x = x[0]
            x = tf.squeeze(x)
            if x.shape[-1] == 3:
                x = x[..., -1]
            temp_y = []
            temp_pred = []
            x_write = self.scale_0_1(tf.cast(self.return_proper_size(x), 'float32'))
            for val in range(len(y_base)):
                y = tf.squeeze(y_base[val])
                index = None
                if len(y.shape) > 2:
                    indexes = tf.unique(tf.where(y > 0)[..., 0])[0]
                    index = indexes[tf.shape(indexes)[0] // 2]
                    y = y[index]
                pred = pred_base[val]
                pred = tf.squeeze(tf.argmax(pred, axis=-1))
                x_write = x
                if index is not None:
                    pred = pred[index]
                    x_write = x[index]
                x_write = self.scale_0_1(tf.cast(self.return_proper_size(x_write), 'float32'))
                pred = self.scale_0_1(tf.cast(self.return_proper_size(pred),'float32'))
                temp_pred.append(pred)
                temp_y.append(self.scale_0_1(tf.cast(self.return_proper_size(y),'float32')))
            pred_out = tf.concat(temp_pred, axis=1)
            y_out = tf.concat(temp_y, axis=1)
            pred_out = tf.image.resize_with_crop_or_pad(pred_out, target_height=512*len(y_base), target_width=512)
            y_out = tf.image.resize_with_crop_or_pad(y_out, target_height=512*len(y_base), target_width=512)
            x_write = tf.image.resize_with_crop_or_pad(x_write, target_height=512, target_width=512)
            output_x.append(x_write)
            output_y.append(y_out)
            output_pred.append(pred_out)
        x, y, pred_out = tf.concat(output_x, axis=2), tf.concat(output_y, axis=2), tf.concat(output_pred, axis=2)
        return x, y, pred_out

    def on_epoch_end(self, epoch, logs=None):
        if self.add_images:
            x, y, pred_out = self.write_images()
            with self.file_writer.as_default():
                tf.summary.image('Image', tf.cast(x, 'float32'), step=epoch)
                tf.summary.image('Truth', tf.cast(y, 'float32'), step=epoch)
                tf.summary.image('Pred', tf.cast(pred_out, 'float32'), step=epoch)
                tf.summary.scalar('Learning_Rate',tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)
        else:
            with self.file_writer.as_default():
                tf.summary.scalar('Learning_Rate',tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)


class MeanDSC(tf.keras.metrics.MeanIoU):
    '''
    This varies from the original in that we don't care about the background DSC
    '''
    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))-1 # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries)
        return tf.math.divide_no_nan(tf.multiply(tf.cast(2,'float32'),jaccard),tf.add(tf.cast(1,'float32'),jaccard),name='mean_dsc')


class MeanJaccard(tf.keras.metrics.MeanIoU):
    '''
    This varies from the original in that we don't care about the background DSC
    '''
    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))-1 # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries, name='mean_jaccard')
        return jaccard


class Base_To_Sparse(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, self._dtype)
        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None and sample_weight.shape.ndims > 1:
            sample_weight = tf.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype='float64')
        return self.total_cm.assign_add(current_cm)


class SparseCategoricalMeanDSC(Base_To_Sparse):
    '''
    This varies from the original in that we don't care about the background DSC
    '''
    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))-1 # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries)
        return tf.math.divide_no_nan(tf.multiply(tf.cast(2,'float32'),jaccard),tf.add(tf.cast(1,'float32'),jaccard),name='mean_dsc')


class SparseCategoricalMeanJaccard(Base_To_Sparse):
    '''
    This varies from the original in that we don't care about the background DSC
    '''
    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))-1 # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries, name='mean_jaccard')
        return jaccard


if __name__ == '__main__':
    pass
