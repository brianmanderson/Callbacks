from keras.callbacks import TensorBoard
import keras.backend as K
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from PIL import Image
import io

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
                 update_freq='epoch', tag='', data_generator=None,image_frequency=5, num_images=3, conv_names=None,
                 write_images=True):
        super().__init__(log_dir=log_dir,
                 histogram_freq=0,
                 batch_size=32,
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
        # out_image, out_truth, out_pred = x[0,...], y[0,...,-1], pred[0,...,-1]
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Image', image=self.make_image(out_image, min_val=np.min(out_image),max_val=np.max(out_image)))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Ground_Truth', image=self.make_image(out_truth,min_val=np.min(out_truth),max_val=np.max(out_truth)))])
        self.writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'Prediction', image=self.make_image(out_pred,min_val=np.min(out_truth),max_val=np.max(out_truth)))])
        self.writer.add_summary(summary, epoch)
        return None


def visualize_model_tensorboard(model ,tensorboard_output):
    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)
    tensorboard = TensorBoard(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
                              write_images=True, update_freq='epoch', histogram_freq=0)
    tensorboard.set_model(model)
    tensorboard._write_logs({}, 0)
    return None


def make_grid_from_activation(layer_activation):
    n_features = layer_activation.shape[-1]
    split = 2
    while n_features / split % 2 == 0 and n_features / split >= split:
        split *= 2
    split /= 2
    images_per_row = int(n_features // split)
    if len(layer_activation.shape) == 4:
        rows_size = layer_activation.shape[1]
        cols_size = layer_activation.shape[2]
    else:
        rows_size = layer_activation.shape[0]
        cols_size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((rows_size * images_per_row, n_cols * cols_size))
    for col in range(n_cols):
        for row in range(images_per_row):
            if len(layer_activation.shape) == 4:
                channel_image = layer_activation[layer_activation.shape[0] // 2, :, :, col * images_per_row + row]
            else:
                channel_image = layer_activation[:, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * rows_size: (row + 1) * rows_size,
            col * cols_size: (col + 1) * cols_size] = channel_image
    return display_grid


class visualization_model_class(object):
    def __init__(self, model,desired_layer_names=None, save_images=False):
        self.save_images = save_images
        self.out_path = None
        all_layers = model.layers[:]
        all_layers = [layer for layer in all_layers if layer.name.find('mask') == -1 and
                      layer.name.lower().find('input') == -1 and
                      layer.name.lower().find('batch_normalization') == -1 and
                      layer.name.lower().find('activation') == -1]
        if desired_layer_names:
            all_layers = [layer for layer in all_layers if layer.name in desired_layer_names]
        self.layer_outputs = [layer.output for layer in all_layers]  # We already have the input.
        self.layer_names = [layer.name for layer in all_layers]  #
        self.activation_model = Model(inputs=model.input, outputs=self.layer_outputs)

    def predict_on_tensor(self, img_tensor):
        self.activations = self.activation_model.predict(img_tensor)

    def define_output(self,out_path):
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def plot_activations(self):
        if not self.out_path and self.save_images:
            self.define_output(os.path.join('.','activation_outputs'))
        image_index = 0
        print(self.layer_names)
        for layer_name, layer_activation in zip(self.layer_names, self.activations):
            print(layer_name)
            print(self.layer_names.index(layer_name) / len(self.layer_names) * 100)
            layer_activation = np.squeeze(layer_activation)
            display_grid = make_grid_from_activation(layer_activation)
            scale = 0.05
            plt.figure(figsize=(display_grid.shape[1] * scale, scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='gray')
            if self.save_images:
                plt.savefig(os.path.join(self.out_path, str(image_index) + '_' + layer_name + '.png'))
                plt.close()
            image_index += 1

def visualize_model(model, img_tensor, out_path = os.path.join('.','activation_outputs')):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # layer_outputs = [layer.output for layer in model.layers]
    # activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    all_layers = model.layers[1:]
    all_layers = [layer for layer in all_layers if layer.name.find('mask') == -1 and layer.name.lower().find('input') == -1 and layer.name.lower().find('batch_normalization') == -1]
    layer_outputs = [layer.output for layer in all_layers]  # We already have the input.
    layer_names = [layer.name for layer in all_layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    image_index = 0
    for layer_name, layer_activation in zip(layer_names, activations):
        print(layer_name)
        print(layer_names.index(layer_name)/len(layer_names) * 100)
        layer_activation = np.squeeze(layer_activation)
        display_grid = make_grid_from_activation(layer_activation)
        scale = 0.05
        plt.figure(figsize=(display_grid.shape[1] * scale, scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')
        plt.savefig(os.path.join(out_path,str(image_index) + '_' + layer_name + '.png'))
        plt.close()
        image_index += 1


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)


def decay_regularization(img, grads, decay = 0.9):
    return decay * img


def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img):
    loss_value, grads_value = loss_function([img])
    gradient_ascent_step = img + grads_value * 0.9

    # Convert to row major format for using opencv routines
    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    # List of regularization functions to use
    regularizations = [decay_regularization, clip_weak_pixel_regularization]

    # The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    # Convert image back to 1 x 3 x height x width
    img = np.float32([np.transpose(img, (2, 0, 1))])

    return img


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[..., filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    img = np.random.random((1, size, size, 3)) * 20 + 128.
    for i in range(30):
        img = gradient_ascent_iteration(iterate, img)
    return deprocess_image(img[0])

def visualize_filters(model):
    layer_name = 'block1_conv1'
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3) ,dtype='uint8')
    for i in range(8):
        print(i)
        for j in range(8):
            filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
