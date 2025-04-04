__author__ = 'Brian M Anderson'

# Created on 4/15/2020
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os
import numpy as np
from tensorflow.keras.metrics import Metric
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from PlotScrollNumpyArrays import plot_scroll_Image
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix


class Add_Images_and_LR(Callback):
    def __init__(self, log_dir, add_images=True, validation_data=None, number_of_images=3, image_frequency=1,
                 threshold_x=False, target_image_height=512, target_image_width=512, arg_max=False):
        super(Add_Images_and_LR, self).__init__()
        self.target_image_height = target_image_height
        self.target_image_width = target_image_width
        self.image_frequency = image_frequency
        self.threshold_x = threshold_x
        if add_images and validation_data is None:
            AssertionError('Need to provide validation data if you want images!')
        self.add_images = add_images
        self.number_of_images = number_of_images
        self.validation_data = validation_data
        if validation_data is not None:
            self.validation_data = iter(validation_data)
        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val_images'))
        self.arg_max = arg_max
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
        if val.shape[0] != 1:
            val = tf.expand_dims(val, axis=0)
        if val.shape[-1] != 1:
            val = tf.expand_dims(val, axis=-1)
        return val

    def scale_0_1(self, val):
        val = tf.subtract(val, tf.reduce_min(val))
        val = tf.divide(val, tf.reduce_max(val))
        return val

    def write_images(self):
        out_dict = {}
        print('Writing out images')
        for i in range(self.number_of_images):
            x, y_base = next(self.validation_data)
            if self.arg_max:
                y_base = tf.argmax(y_base, axis=-1)
            print('Writing out image {}'.format(i))
            # x, y_base = self.image_dict[i]
            y = y_base
            start_index = None
            while type(y) is tuple:
                y = y[0]
            y = tf.squeeze(y)
            if len(y.shape) > 2 and y.shape[0] > 128:
                indexes = tf.unique(tf.where(y > 0)[..., 0])[0]
                start_index = indexes[indexes.shape[0] // 2]
            if start_index is not None:
                x_pred = []
                if type(x) is tuple:
                    for i in x:
                        expand_0 = i.shape[0] == 1
                        end_shape = i.shape[-1]
                        i = tf.squeeze(i)
                        i = i[tf.maximum(start_index - 16, 0):tf.minimum(start_index + 16, i.shape[0])]
                        if expand_0:
                            i = tf.expand_dims(i, axis=0)
                        if end_shape == 1:
                            i = tf.expand_dims(i, axis=-1)
                        x_pred.append(i)
                    x = tuple(x_pred)
                y_pred = []
                if type(y_base) is tuple:
                    for i in y_base:
                        expand_0 = i.shape[0] == 1
                        end_shape = i.shape[-1]
                        i = tf.squeeze(i)
                        i = i[tf.maximum(start_index - 16, 0):tf.minimum(start_index + 16, i.shape[0])]
                        if expand_0:
                            i = tf.expand_dims(i, axis=0)
                        if end_shape == 1:
                            i = tf.expand_dims(i, axis=-1)
                        y_pred.append(i)
                    y_base = tuple(y_pred)
            pred_base = self.model(x, training=False)
            while type(x) is tuple:
                x = x[0]
            x = tf.squeeze(x)
            if x.shape[-1] == 3:
                x = x[..., -1]
            if self.threshold_x:
                x = tf.where(x > 5, 5, x)
                x = tf.where(x < -5, -5, x)
            for val in range(len(y_base)):
                y = tf.squeeze(y_base[val])
                index = None
                if len(y.shape) > 2:
                    y_temp = tf.where(y > 0)[..., 0]
                    if y_temp.shape.dims[0] == 0:
                        y_temp = tf.where(y >= 0)[..., 0]
                    indexes = tf.unique(y_temp)[0]
                    index = indexes[tf.shape(indexes)[0] // 2]
                    y = y[index]
                if type(pred_base) is tuple:
                    pred = pred_base[val]
                else:
                    pred = pred_base
                pred = tf.squeeze(tf.argmax(pred, axis=-1))[val]
                if len(y.shape) > 2:
                    y = tf.argmax(y, axis=-1)
                x_write = x[val]
                if index is not None:
                    pred = pred[index]
                    x_write = x[index]
                x_write = self.scale_0_1(tf.cast(self.return_proper_size(x_write), 'float32')) * 255
                x_write = tf.image.resize_with_crop_or_pad(x_write, target_height=self.target_image_height,
                                                           target_width=self.target_image_width)
                pred_write = self.scale_0_1(tf.cast(self.return_proper_size(pred), 'float32')) * 255
                pred_write = tf.image.resize_with_crop_or_pad(pred_write, target_height=self.target_image_height,
                                                              target_width=self.target_image_width)
                y_write = self.scale_0_1(tf.cast(self.return_proper_size(y), 'float32')) * 255
                y_write = tf.image.resize_with_crop_or_pad(y_write, target_height=self.target_image_height,
                                                           target_width=self.target_image_width)
                image = tf.concat([x_write, y_write, pred_write], axis=1)
                if val not in out_dict:
                    out_dict[val] = image
                else:
                    out_dict[val] = tf.concat([out_dict[val], image], axis=2)
        outputs = []
        for i in out_dict.keys():
            outputs.append(out_dict[i])
        del out_dict
        return outputs

    def on_epoch_end(self, epoch, logs=None):
        if self.add_images and self.image_frequency != 0 and epoch % self.image_frequency == 0:
            outputs = self.write_images()
            with self.file_writer.as_default():
                for i, output in enumerate(outputs):
                    tf.summary.image('Image Truth Pred {}'.format(i), tf.cast(output, 'float32'), step=epoch)
                tf.summary.scalar('Learning_Rate', tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)
        else:
            with self.file_writer.as_default():
                tf.summary.scalar('Learning_Rate', tf.keras.backend.get_value(self.model.optimizer.lr), step=epoch)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='dice_loss', image_shape=(32, 512, 512), d_type='float32'):
        self.image_shape = image_shape
        super(DiceLoss, self).__init__(name=name)
        self._dtype = d_type

    def call(self, y_true, y_pred):
        spatial_axes = tuple(range(1, len(self.image_shape) - 1))
        # Just take the outcome prediction
        y_true = y_true[..., 1]
        y_true = tf.cast(y_true, self._dtype)
        y_pred = y_pred[..., 1]

        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=spatial_axes)
        denominator = tf.reduce_sum(y_true + y_pred, axis=spatial_axes)

        dice_score = numerator / denominator
        # To handle the numerical stability and prevent division by zero
        dice_loss = 1 - tf.math.divide_no_nan(dice_score, tf.reduce_sum(dice_score))

        return dice_loss


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='dice_loss', image_shape=(32, 512, 512), d_type='float32'):
        self.image_shape = image_shape
        super(DiceLoss, self).__init__(name=name)
        self._dtype = d_type

    def call(self, y_true, y_pred):
        spatial_axes = tuple(range(1, len(self.image_shape) - 1))
        # Just take the outcome prediction
        y_true = y_true[..., 1]
        y_true = tf.cast(y_true, self._dtype)
        y_pred = y_pred[..., 1]


        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=spatial_axes)
        denominator = tf.reduce_sum(y_true + y_pred, axis=spatial_axes)
        
        dice_score = numerator / denominator
        # To handle the numerical stability and prevent division by zero
        dice_loss = 1 - tf.math.divide_no_nan(dice_score, tf.reduce_sum(dice_score))

        return dice_loss


class MeanDSC(tf.keras.metrics.MeanIoU):
    '''
    This varies from the original in that we don't care about the background DSC
    '''

    def __init__(self, num_classes, name='mean_dsc', dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(MeanDSC, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

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

        y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, self._dtype)
        # Flatten the input if its rank > 1.
        y_pred = tf.reshape(y_pred, [-1])

        y_true = tf.reshape(y_true, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype='float32')
        return self.total_cm.assign_add(current_cm)

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
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)) - 1  # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.cast(tf.reduce_sum(iou), tf.float32), tf.cast(num_valid_entries, tf.float32))
        return tf.math.divide_no_nan(tf.multiply(tf.cast(2, 'float32'), jaccard),
                                     tf.add(tf.cast(1, 'float32'), jaccard), name='mean_dsc')

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanDSC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReducedMeanDSC(MeanDSC):
    '''
    This varies from the original in that we don't care about some classes
    '''

    def __init__(self, classes_of_interest=None, name='mean_dsc_reduced', dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          classes_of_interest: List of class indexes that you want to calculate the DSC on, example [0, 1, 2, 4, 6]
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        assert type(classes_of_interest) is tuple or list, 'Provide a list of classes you want the dice on'
        assert classes_of_interest[0] == 0, "You'll still want the background, as it goes by argmax"
        self.classes_of_interest = classes_of_interest
        num_classes = len(classes_of_interest)
        super(ReducedMeanDSC, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

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
        y_true = tf.argmax(tf.stack([y_true[..., i] for i in self.classes_of_interest], axis=-1), axis=-1)
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.argmax(tf.stack([y_pred[..., i] for i in self.classes_of_interest], axis=-1), axis=-1)
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

    def get_config(self):
        config = {'num_classes': self.num_classes, 'classes_of_interest': self.classes_of_interest}
        base_config = super(ReducedMeanDSC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanJaccard(tf.keras.metrics.MeanIoU):
    '''
    This varies from the original in that we don't care about the background DSC
    '''

    def __init__(self, num_classes, name='mean_jaccard', dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(MeanJaccard, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

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

        y_true = tf.argmax(y_true, axis=-1)
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
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)) - 1  # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries, name='mean_jaccard')
        return jaccard

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(MeanJaccard, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseCategoricalMeanDSC(Metric):
    '''
    This varies from the original in that we don't care about the background DSC
    '''

    def __init__(self, num_classes, name='sparse_categorical_mean_dsc', dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(SparseCategoricalMeanDSC, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)

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
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)) - 1  # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries)
        return tf.math.divide_no_nan(tf.multiply(tf.cast(2, 'float32'), jaccard),
                                     tf.add(tf.cast(1, 'float32'), jaccard), name='mean_dsc')

    def reset_state(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(SparseCategoricalMeanDSC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseCategoricalMeanJaccard(Metric):
    '''
    This varies from the original in that we don't care about the background DSC
    '''

    def __init__(self, num_classes, name='sparse_categorical_mean_jaccard', dtype=None):
        """Creates a `MeanIoU` instance.

        Args:
          num_classes: The possible number of labels the prediction task can have.
            This value must be provided, since a confusion matrix of dimension =
            [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(SparseCategoricalMeanJaccard, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64)

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
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)) - 1  # pitch out background

        iou = tf.math.divide_no_nan(true_positives, denominator)[1:]

        jaccard = tf.math.divide_no_nan(tf.reduce_sum(iou), num_valid_entries, name='mean_jaccard')
        return jaccard

    def reset_state(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(SparseCategoricalMeanJaccard, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    pass
