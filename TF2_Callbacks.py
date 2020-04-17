__author__ = 'Brian M Anderson'

# Created on 4/15/2020
import tensorflow as tf
import os
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest


def _tpu_multi_host_concat(v, strategy):
    """Correctly order TPU PerReplica objects."""
    replicas = strategy.unwrap(v)
    # When distributed datasets are created from Tensors / NumPy,
    # TPUStrategy.experimental_distribute_dataset shards data in
    # (Replica, Host) order, and TPUStrategy.unwrap returns it in
    # (Host, Replica) order.
    # TODO(b/150317897): Figure out long-term plan here.
    num_replicas_per_host = strategy.extended.num_replicas_per_host
    ordered_replicas = []
    for replica_id in range(num_replicas_per_host):
        ordered_replicas += replicas[replica_id::num_replicas_per_host]
    return concat(ordered_replicas)


def concat(tensors, axis=0):
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], sparse_tensor.SparseTensor):
        return sparse_ops.sparse_concat_v2(axis=axis, sp_inputs=tensors)
    if isinstance(tensors[0], ragged_tensor.RaggedTensor):
        return ragged_concat_ops.concat(tensors, axis=axis)
    return array_ops.concat(tensors, axis=axis)


def _is_tpu_multi_host(strategy):
    return (dist_utils.is_tpu_strategy(strategy) and
            strategy.extended.num_hosts > 1)


def reduce_per_replica(values, strategy, reduction='first'):
    """Reduce PerReplica objects.

  Arguments:
    values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
      returned as-is.
    strategy: `tf.distribute.Strategy` object.
    reduction: One of 'first', 'concat'.

  Returns:
    Structure of `Tensor`s.
  """

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if not isinstance(v, ds_values.PerReplica):
            return v
        elif reduction == 'first':
            return strategy.unwrap(v)[0]
        elif reduction == 'concat':
            if _is_tpu_multi_host(strategy):
                return _tpu_multi_host_concat(v, strategy)
            else:
                return concat(strategy.unwrap(v))
        else:
            raise ValueError('`reduction` must be "first" or "concat".')

    return nest.map_structure(_reduce, values)


class Add_Images(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, number_of_images=3):
        super(Add_Images, self).__init__()
        self.validation_data = iter(validation_data)
        # self.val_x, self.val_y, self.val_sample_weight = (data_adapter.unpack_x_y_sample_weight(validation_data))
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

    def on_epoch_end_test(self, epoch, logs=None):
        data_handler = data_adapter.DataHandler(
            x=self.val_x,
            y=self.val_y,
            sample_weight=self.val_sample_weight,
            batch_size=None,
            steps_per_epoch=10,
            initial_epoch=0,
            epochs=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            model=self)
        for _, iterator in data_handler.enumerate_epochs():
            data = next(iterator)
            outputs = self.distribute_strategy.run(self.test_step, args=(data,))
            outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction='first')
            xxx = 1

    def on_epoch_end(self, epoch, logs=None):
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
        with self.file_writer.as_default():
            tf.summary.image('Image', tf.cast(x, 'float32'), step=epoch)
            tf.summary.image('Truth', tf.cast(y, 'float32'), step=epoch)
            tf.summary.image('Pred', tf.cast(pred_out, 'float32'), step=epoch)


if __name__ == '__main__':
    pass
