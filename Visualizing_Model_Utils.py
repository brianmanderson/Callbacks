from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks_v1 import TensorBoard
from tensorflow.python.keras import backend as K
import numpy as np
import os, six
import tensorflow.compat.v1 as tf
from PIL import Image
import io
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables, array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.backend import get_value
from .Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


class TensorBoardImage(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,is_segmentation=True,
                 write_graph=True,
                 write_grads=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,save_dir = None,batch_steps=None,
                 update_freq='epoch', tag='', data_generator=None,
                 image_frequency=5, num_images=3, conv_names=None,
                 write_images=True, profile_batch=0):
        super().__init__(log_dir=log_dir,
                 histogram_freq=0,
                 batch_size=32,profile_batch=profile_batch,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch')
        self.write_images = write_images
        self.conv_names = conv_names
        self.is_segmentation = is_segmentation
        if batch_steps:
            self.epoch_index = 0
        else:
            self.epoch_index = None
        self.save_dir = save_dir
        self.image_frequency = image_frequency
        self.num_images = num_images
        self.tag = tag
        self.log_dir = log_dir
        self.data_generator = data_generator
        if self.data_generator:
            x, y = self.data_generator.__getitem__(0)
            if type(x) == list:
                x = x[0]
            if len(x.shape) == 5:
                self.images_x, self.rows_x, self.cols_x = x.shape[1:4]
                self.images_y, self.rows_y, self.cols_y = y.shape[1:4]
            else:
                self.images_x, self.rows_x, self.cols_x = x.shape[:3]
                self.images_y, self.rows_y, self.cols_y = y.shape[:3]
            self.classes = y.shape[-1]
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_grads = write_grads
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            # if name.startswith('val_'):
            #     writer = self.val_writer
            # else:
            #     writer = self.writer
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()
        # self.val_writer.flush()

    def make_image(self, tensor, min_val, max_val):
        """
        Convert an numpy representation image to Image protobuf.
        Copied from https://github.com/lanpa/tensorboard-pytorch/
        """
        print([min_val,max_val])
        tensor = np.squeeze(tensor)
        height, width = tensor.shape
        tensor = ((tensor - min_val) / (max_val-min_val) * 255).astype('uint8')
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=1,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.data_generator and epoch % self.image_frequency == 0 and not self.epoch_index: # If we're doing batch, leave it,  and np.max(logs['val_dice_coef_3D'])>0.2
            self.data_generator.on_epoch_end()
            if self.write_images:
                self.add_images(epoch, self.num_images)
            if self.conv_names is not None:
                self.add_conv(epoch)



        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {_input: embeddings_data[idx][batch]
                                     for idx, _input in enumerate(self.model.input)}
                    else:
                        feed_dict = {self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def add_conv(self, epoch):
        layer_names = [i.name for i in self.model.layers]
        for conv_name in self.conv_names:
            weights = self.model.layers[layer_names.index(conv_name)].get_weights()[0][:, :, 0, :]
            n_features = weights.shape[-1]
            split = 2
            while n_features / split % 2 == 0 and n_features / split >= split:
                split *= 2
            split /= 2
            images_per_row = int(n_features // split)
            if len(weights.shape) == 4:
                rows_size = weights.shape[1]
                cols_size = weights.shape[2]
            else:
                rows_size = weights.shape[0]
                cols_size = weights.shape[1]
            n_cols = n_features // images_per_row
            out_image = np.ones(
                (rows_size * images_per_row + images_per_row - 1, n_cols * cols_size + n_cols - 1)) * np.min(weights)
            step = 0
            for col in range(n_cols):
                for row in range(images_per_row):
                    weight = weights[..., step]
                    weight = (weight - np.mean(weight)) / np.std(weight)
                    out_image[row + row * rows_size:row + (row + 1) * rows_size,
                    col + col * cols_size:col + (col + 1) * cols_size] = weight
                    step += 1
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+conv_name, image=self.make_image(out_image, min_val=np.min(out_image),max_val=np.max(out_image)))])
            self.writer.add_summary(summary, epoch)
        return None

    def add_images(self, epoch, num_images=3):
        # Load image
        print('Adding images')
        gap_x, gap_y = int(self.cols_x/10), int(self.cols_y/10)
        out_image, out_truth, out_pred = np.zeros([self.rows_x, int(self.cols_x*num_images+gap_x*(num_images-1))]),\
                                         np.zeros([self.rows_y, int(self.cols_y*num_images+gap_y*(num_images-1))]),\
                                         np.zeros([self.rows_y, int(self.cols_y * num_images + gap_y * (num_images - 1))])
        image_indexes = np.asarray(range(len(self.data_generator)))
        np.random.shuffle(image_indexes)
        for i in range(num_images):
            if len(image_indexes) < i:
                continue
            image_index = image_indexes[i]
            start_x, start_y = int(gap_x * i), int(gap_y * i)
            print(i)
            x, y = self.data_generator.__getitem__(image_index)
            pred = self.model.predict(x)
            if type(x) == list:
                x = x[0]
            x = np.squeeze(x)
            y = np.squeeze(y)
            pred = np.squeeze(pred)
            if x.shape[-1] == 3:
                x = x[...,0]
            if len(x.shape)==3:
                rows, cols = x.shape[1], x.shape[2]
            elif len(x.shape) == 2:
                rows, cols = x.shape[0], x.shape[1]
                y = y[None,...]
                pred = pred[None,...]
            else:
                rows, cols = self.rows_x, self.cols_x
            if self.is_segmentation:
                pred = np.argmax(pred, axis=-1)[...,None]
                y = np.argmax(y, axis=-1)[...,None]
                slices = np.where(np.max(y, axis=tuple([i for i in range(1,len(y.shape))])) != 0)[0]
                index = slices[len(slices) // 2]
            else:
                index = x.shape[0]//2

            if rows != self.rows_x or cols != self.cols_x:
                num_images = 1
                self.rows_x = self.rows_y = rows
                self.cols_x = self.cols_y = cols
                gap_x, gap_y = int(self.cols_x / 10), int(self.cols_y / 10)
                out_image, out_truth, out_pred = np.zeros(
                    [self.rows_x, int(self.cols_x * num_images + gap_x * (num_images - 1))]), \
                                                 np.zeros([self.rows_y,
                                                           int(self.cols_y * num_images + gap_y * (num_images - 1))]), \
                                                 np.zeros([self.rows_y,
                                                           int(self.cols_y * num_images + gap_y * (num_images - 1))])
                out_image[...] = x[index,...]
                out_truth[...] = y[index,..., -1]
                out_pred[...] = pred[index,..., -1]
                break
            else:
                if len(x.shape) == 4:
                    x = x[...,-1]
                out_image[:, self.cols_x * i + start_x:self.cols_x * (i + 1) + start_x] = x[index,...]
                out_truth[:, self.cols_y * i + start_y:self.cols_y * (i + 1) + start_y] = y[index,..., -1]
                out_pred[:, self.cols_y * i + start_y:self.cols_y * (i + 1) + start_y] = pred[index,..., -1]
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.save(os.path.join(self.save_dir,'Out_Image_' + str(epoch) + '.npy'),out_image)
            np.save(os.path.join(self.save_dir, 'Out_Truth_' + str(epoch) + '.npy'), out_truth)
            np.save(os.path.join(self.save_dir, 'Out_Pred_' + str(epoch) + '.npy'), out_pred)
        print(out_image.shape)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Image', image=self.make_image(out_image, min_val=np.min(out_image),max_val=np.max(out_image)))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Ground_Truth', image=self.make_image(out_truth,min_val=np.min(out_truth),max_val=np.max(out_truth)))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Prediction', image=self.make_image(out_pred,min_val=np.min(out_truth),max_val=np.max(out_truth)))])
        self.writer.add_summary(summary, epoch)
        return None


class TensorBoardImage_v2(Callback):
    def __init__(self, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch',
                 profile_batch=2, embeddings_freq=0, embeddings_metadata=None,
                 image_frequency=5, data_generator=None, num_images=3, conv_names=None,
                 is_segmentation=True, save_dir=None, **kwargs):
        super(TensorBoardImage, self).__init__()
        self._validate_kwargs(kwargs)
        self.conv_names = conv_names
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        if update_freq == 'batch':
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata

        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._current_batch = 0

        # A collection of file writers currently in use, to be closed when
        # training ends for this callback. Writers are keyed by the
        # directory name under the root logdir: e.g., "train" or
        # "validation".
        self._train_run_name = 'train'
        self._validation_run_name = 'validation'
        self._writers = {}

        self._profile_batch = profile_batch
        # True when a trace is running.
        self._is_tracing = False

        # TensorBoard should only write summaries on the chief when in a
        # Multi-Worker setting.
        self._chief_worker_only = True
        self.save_dir = save_dir
        self.data_generator = data_generator
        self.image_frequency = image_frequency
        self.num_images = num_images
        self.write_images = write_images
        self.is_segmentation = is_segmentation
        if data_generator is not None and self.write_images:
            x, y = self.data_generator.__getitem__(0)
            if type(x) == list:
                x = x[0]
            x = np.squeeze(x)
            if x.shape[-1] == 3:
                x = x[...,0]
            y = np.squeeze(y)
            if self.is_segmentation:
                y = np.argmax(y,axis=-1)
            if len(x.shape) == 5:
                self.images_x, self.rows_x, self.cols_x = x.shape[1:4]
                self.images_y, self.rows_y, self.cols_y = y.shape[1:4]
            else:
                self.images_x, self.rows_x, self.cols_x = x.shape[:3]
                self.images_y, self.rows_y, self.cols_y = y.shape[:3]

    def add_conv(self, epoch):
        layer_names = [i.name for i in self.model.layers]
        for conv_name in self.conv_names:
            weights = self.model.layers[layer_names.index(conv_name)].get_weights()[0][:, :, 0, :]
            n_features = weights.shape[-1]
            split = 2
            while n_features / split % 2 == 0 and n_features / split >= split:
                split *= 2
            split /= 2
            images_per_row = int(n_features // split)
            if len(weights.shape) == 4:
                rows_size = weights.shape[1]
                cols_size = weights.shape[2]
            else:
                rows_size = weights.shape[0]
                cols_size = weights.shape[1]
            n_cols = n_features // images_per_row
            out_image = np.ones(
                (rows_size * images_per_row + images_per_row - 1, n_cols * cols_size + n_cols - 1)) * np.min(weights)
            step = 0
            for col in range(n_cols):
                for row in range(images_per_row):
                    weight = weights[..., step]
                    weight = (weight - np.mean(weight)) / np.std(weight)
                    out_image[row + row * rows_size:row + (row + 1) * rows_size,
                    col + col * cols_size:col + (col + 1) * cols_size] = weight
                    step += 1
            summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+conv_name, image=self.make_image(out_image, min_val=np.min(out_image),max_val=np.max(out_image)))])
            self.writer.add_summary(summary, epoch)
        return None

    def add_images(self, epoch, num_images=3):
        # Load image
        print('Adding images')
        gap_x, gap_y = int(self.cols_x/10), int(self.cols_y/10)
        out_image, out_truth, out_pred = np.zeros([self.rows_x, int(self.cols_x*num_images+gap_x*(num_images-1))]),\
                                         np.zeros([self.rows_y, int(self.cols_y*num_images+gap_y*(num_images-1))]),\
                                         np.zeros([self.rows_y, int(self.cols_y * num_images + gap_y * (num_images - 1))])
        image_indexes = np.asarray(range(len(self.data_generator)))
        np.random.shuffle(image_indexes)
        for i in range(num_images):
            if len(image_indexes) < i:
                continue
            image_index = image_indexes[i]
            start_x, start_y = int(gap_x * i), int(gap_y * i)
            print(i)
            x, y = self.data_generator.__getitem__(image_index)
            pred = self.model.predict(x)
            if type(x) == list:
                x = x[0]
            x = np.squeeze(x)
            y = np.squeeze(y)
            pred = np.squeeze(pred)
            if x.shape[-1] == 3:
                x = x[...,0]
            if len(x.shape)==3:
                rows, cols = x.shape[1], x.shape[2]
            elif len(x.shape) == 2:
                rows, cols = x.shape[0], x.shape[1]
                y = y[None,...]
                pred = pred[None,...]
            else:
                rows, cols = self.rows_x, self.cols_x
            if self.is_segmentation:
                pred = np.argmax(pred, axis=-1)[...,None]
                y = np.argmax(y, axis=-1)[...,None]
                slices = np.where(np.max(y, axis=tuple([i for i in range(1,len(y.shape))])) != 0)[0]
                index = slices[len(slices) // 2]
            else:
                index = x.shape[0]//2

            if rows != self.rows_x or cols != self.cols_x:
                num_images = 1
                self.rows_x = self.rows_y = rows
                self.cols_x = self.cols_y = cols
                gap_x, gap_y = int(self.cols_x / 10), int(self.cols_y / 10)
                out_image, out_truth, out_pred = np.zeros(
                    [self.rows_x, int(self.cols_x * num_images + gap_x * (num_images - 1))]), \
                                                 np.zeros([self.rows_y,
                                                           int(self.cols_y * num_images + gap_y * (num_images - 1))]), \
                                                 np.zeros([self.rows_y,
                                                           int(self.cols_y * num_images + gap_y * (num_images - 1))])
                out_image[...] = x[index,...]
                out_truth[...] = y[index,..., -1]
                out_pred[...] = pred[index,..., -1]
                break
            else:
                if len(x.shape) == 4:
                    x = x[...,-1]
                out_image[:, self.cols_x * i + start_x:self.cols_x * (i + 1) + start_x] = x[index,...]
                out_truth[:, self.cols_y * i + start_y:self.cols_y * (i + 1) + start_y] = y[index,..., -1]
                out_pred[:, self.cols_y * i + start_y:self.cols_y * (i + 1) + start_y] = pred[index,..., -1]
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            np.save(os.path.join(self.save_dir,'Out_Image_' + str(epoch) + '.npy'),out_image)
            np.save(os.path.join(self.save_dir, 'Out_Truth_' + str(epoch) + '.npy'), out_truth)
            np.save(os.path.join(self.save_dir, 'Out_Pred_' + str(epoch) + '.npy'), out_pred)
        print(out_image.shape)
        out_image = ((out_image - np.min(out_image)) / (np.max(out_image) - np.min(out_image)) * 255).astype('uint8')
        out_pred = ((out_pred - np.min(out_truth)) / (np.max(out_truth) - np.min(out_truth)) * 255).astype('uint8')
        out_truth = ((out_truth - np.min(out_truth)) / (np.max(out_truth) - np.min(out_truth)) * 255).astype('uint8')
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(),  writer.as_default(),  summary_ops_v2.always_record_summaries():
            for image, name in zip([out_image,out_pred,out_truth],['Image','Prediction','Ground_Truth']):
                w_img = array_ops.squeeze(image)
                shape = K.int_shape(w_img)
                w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
                summary_ops_v2.image(name, w_img, step=epoch)
            writer.flush()
        return None

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        if self.write_images and self.data_generator and epoch % self.image_frequency == 0: # If we're doing batch, leave it,  and np.max(logs['val_dice_coef_3D'])>0.2
            self.data_generator.on_epoch_end()
            if self.write_images:
                self.add_images(epoch, self.num_images)
            if self.conv_names is not None:
                self.add_conv(epoch)
        if logs is not None:
            logs['val_learning_rate'] = get_value(self.model.optimizer.lr)
        else:
            logs = {'val_learning_rate':get_value(self.model.optimizer.lr)}
        self._log_metrics(logs, prefix='epoch_', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def _validate_kwargs(self, kwargs):
        """Handle arguments were supported in V1."""
        if kwargs.get('write_grads', False):
            logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                            'for the `TensorBoard` Callback.')
        if kwargs.get('batch_size', False):
            logging.warning('`batch_size` is no longer needed in the '
                            '`TensorBoard` Callback and will be ignored '
                            'in TensorFlow 2.0.')
        if kwargs.get('embeddings_layer_names', False):
            logging.warning('`embeddings_layer_names` is not supported in '
                            'TensorFlow 2.0. Instead, all `Embedding` layers '
                            'will be visualized.')
        if kwargs.get('embeddings_data', False):
            logging.warning('`embeddings_data` is not supported in TensorFlow '
                            '2.0. Instead, all `Embedding` variables will be '
                            'visualized.')

        unrecognized_kwargs = set(kwargs.keys()) - {
            'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
        }

        # Only allow kwargs that were supported in V1.
        if unrecognized_kwargs:
            raise ValueError('Unrecognized arguments in `TensorBoard` '
                             'Callback: ' + str(unrecognized_kwargs))

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self.model = model
        with context.eager_mode():
            self._close_writers()
            if self.write_graph:
                with self._get_writer(self._train_run_name).as_default():
                    with summary_ops_v2.always_record_summaries():
                        if not model.run_eagerly:
                            summary_ops_v2.graph(K.get_graph(), step=0)

                        summary_writable = (
                                self.model._is_graph_network or  # pylint: disable=protected-access
                                self.model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
                        if summary_writable:
                            summary_ops_v2.keras_model('keras', self.model, step=0)

        if self.embeddings_freq:
            self._configure_embeddings()

        self._prev_summary_writer = context.context().summary_writer
        self._prev_summary_recording = context.context().summary_recording
        self._prev_summary_step = context.context().summary_step

    def on_train_begin(self, logs=None):
        self._init_batch_steps()
        if self._profile_batch == 1:
            summary_ops_v2.trace_on(graph=True, profiler=True)
            self._is_tracing = True

    def on_test_begin(self, logs=None):
        self._set_default_writer(self._validation_run_name)

    def on_train_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.

        Performs profiling if current batch is in profiler_batches.

        Arguments:
          batch: Integer, index of batch within the current epoch.
          logs: Dict. Metric results for this batch.
        """
        if self.update_freq == 'epoch' and self._profile_batch is None:
            return

        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        train_batches = self._total_batches_seen[self._train_run_name]
        if self.update_freq != 'epoch' and batch % self.update_freq == 0:
            self._log_metrics(logs, prefix='batch_', step=train_batches)

        self._increment_step(self._train_run_name)

        if context.executing_eagerly():
            if self._is_tracing:
                self._log_trace()
            elif (not self._is_tracing and
                  math_ops.equal(train_batches, self._profile_batch - 1)):
                self._enable_trace()

    def on_test_batch_end(self, batch, logs=None):
        if self.update_freq == 'epoch':
            return
        self._increment_step(self._validation_run_name)

    def on_epoch_begin(self, epoch, logs=None):
        self._set_default_writer(self._train_run_name)

    def on_train_end(self, logs=None):
        if self._is_tracing:
            self._log_trace()
        self._close_writers()

        context.context().summary_writer = self._prev_summary_writer
        context.context().summary_recording = self._prev_summary_recording
        context.context().summary_step = self._prev_summary_step

    def _enable_trace(self):
        if context.executing_eagerly():
            summary_ops_v2.trace_on(graph=True, profiler=True)
            self._is_tracing = True

    def _log_trace(self):
        """Logs the trace graph to TensorBoard."""
        if context.executing_eagerly():
            with self._get_writer(self._train_run_name).as_default(), \
                 summary_ops_v2.always_record_summaries():
                # TODO(b/126388999): Remove step info in the summary name.
                step = K.get_value(self._total_batches_seen[self._train_run_name])
                summary_ops_v2.trace_export(
                    name='batch_%d' % step,
                    step=step,
                    profiler_outdir=os.path.join(self.log_dir, 'train'))
            self._is_tracing = False

    def _log_metrics(self, logs, prefix, step):
        """Writes metrics out as custom scalar summaries.

        Arguments:
            logs: Dict. Keys are scalar summary names, values are NumPy scalars.
            prefix: String. The prefix to apply to the scalar summary names.
            step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
            self._train_run_name: [],
            self._validation_run_name: [],
        }
        validation_prefix = 'val_'
        for (name, value) in logs.items():
            if name in ('batch', 'size', 'num_steps'):
                # Scrub non-metric items.
                continue
            if name.startswith(validation_prefix):
                name = name[len(validation_prefix):]
                writer_name = self._validation_run_name
            else:
                writer_name = self._train_run_name
            name = prefix + name  # assign batch or epoch prefix
            logs_by_writer[writer_name].append((name, value))

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Don't create a "validation" events file if we don't
                        # actually have any validation data.
                        continue
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            summary_ops_v2.scalar(name, value, step=step)

    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), \
             writer.as_default(), \
             summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(':', '_')
                    with ops.init_scope():
                        weight = K.get_value(weight)
                    summary_ops_v2.histogram(weight_name, weight, step=epoch)
                    if self.write_images:
                        self._log_weight_as_image(weight, weight_name, epoch)
            writer.flush()

    def _log_weight_as_image(self, weight, weight_name, epoch):
        """Logs a weight as a TensorBoard image."""
        w_img = array_ops.squeeze(weight)
        shape = K.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if K.image_data_format() == 'channels_last':
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

        shape = K.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            summary_ops_v2.image(weight_name, w_img, step=epoch)

    def _log_embeddings(self, epoch):
        embeddings_ckpt = os.path.join(self.log_dir, 'train',
                                       'keras_embedding.ckpt-{}'.format(epoch))
        self.model.save_weights(embeddings_ckpt)

    def _configure_embeddings(self):
        """Configure the Projector for embeddings."""
        # TODO(omalleyt): Add integration tests.
        from tensorflow.python.keras.layers import embeddings
        try:
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                              'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for layer in self.model.layers:
            if isinstance(layer, embeddings.Embedding):
                embedding = config.embeddings.add()
                embedding.tensor_name = layer.embeddings.name

                if self.embeddings_metadata is not None:
                    if isinstance(self.embeddings_metadata, str):
                        embedding.metadata_path = self.embeddings_metadata
                    else:
                        if layer.name in embedding.metadata_path:
                            embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

        if self.embeddings_metadata:
            raise ValueError('Unrecognized `Embedding` layer names passed to '
                             '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                             'argument: ' + str(self.embeddings_metadata.keys()))

        class DummyWriter(object):
            """Dummy writer to conform to `Projector` API."""

            def __init__(self, logdir):
                self.logdir = logdir

            def get_logdir(self):
                return self.logdir

        writer = DummyWriter(self.log_dir)
        projector.visualize_embeddings(writer, config)

    def _close_writers(self):
        """Close all remaining open file writers owned by this callback.

        If there are no such file writers, this is a no-op.
        """
        with context.eager_mode():
            for writer in six.itervalues(self._writers):
                writer.close()
            self._writers.clear()

    def _get_writer(self, writer_name):
        """Get a summary writer for the given subdirectory under the logdir.

        A writer will be created if it does not yet exist.

        Arguments:
          writer_name: The name of the directory for which to create or
            retrieve a writer. Should be either `self._train_run_name` or
            `self._validation_run_name`.

        Returns:
          A `SummaryWriter` object.
        """
        if writer_name not in self._writers:
            path = os.path.join(self.log_dir, writer_name)
            writer = summary_ops_v2.create_file_writer_v2(path)
            self._writers[writer_name] = writer
        return self._writers[writer_name]

    def _set_default_writer(self, writer_name):
        """Sets the default writer for custom batch-level summaries."""
        if self.update_freq == 'epoch':
            # Writer is only used for custom summaries, which are written
            # batch-by-batch.
            return
        writer = self._get_writer(writer_name)
        step = self._total_batches_seen[writer_name]
        context.context().summary_writer = writer

        def _should_record():
            return math_ops.equal(step % self.update_freq, 0)

        context.context().summary_recording = _should_record
        summary_ops_v2.set_step(step)

    def _init_batch_steps(self):
        """Create the total batch counters."""
        if ops.executing_eagerly_outside_functions():
            # Variables are needed for the `step` value of custom tf.summaries
            # to be updated inside a tf.function.
            self._total_batches_seen = {
                self._train_run_name: variables.Variable(0, dtype='int64'),
                self._validation_run_name: variables.Variable(0, dtype='int64')
            }
        else:
            # Custom tf.summaries are not supported in legacy graph mode.
            self._total_batches_seen = {
                self._train_run_name: 0,
                self._validation_run_name: 0
            }

    def _increment_step(self, writer_name):
        step = self._total_batches_seen[writer_name]
        if isinstance(step, variables.Variable):
            step.assign_add(1)
        else:
            self._total_batches_seen[writer_name] += 1


if __name__ == '__main__':
    pass