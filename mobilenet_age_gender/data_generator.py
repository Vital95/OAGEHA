import numpy as np
from random import shuffle

from keras.utils import np_utils

from preprocessor import preprocess_input
from preprocessor import _imread as imread
from preprocessor import _imresize as imresize
import scipy.ndimage as ndi
import cv2


class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations.
    """

    def __init__(self, ground_truth_data, batch_size, image_size,
                 train_keys, validation_keys,
                 path_prefix=None,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 horizontal_flip_probability=0.5,
                 vertical_flip_probability=0.5,
                 do_random_crop=False,
                 grayscale=False,
                 zoom_range=[0.75, 1.25],
                 translation_factor=.3):

        self.ground_truth_data = ground_truth_data
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.grayscale = grayscale
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.zoom_range = zoom_range
        self.translation_factor = translation_factor

    def do_random_rotation(self, image_array):
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                         self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                                                            crop_matrix, offset=offset, order=0, mode='nearest',
                                                            cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array + (1 - alpha) * gray_scale[:, :, None]
        return np.clip(image_array, 0, 255)

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                      np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1, 3) /
                                   255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0, 255)

    def horizontal_flip(self, image_array, box_corners=None):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            if box_corners is not None:
                box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners=None):
        if np.random.random() < self.vertical_flip_probability:
            image_array = image_array[::-1]
            if box_corners is not None:
                box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners

    def transform(self, image_array, box_corners=None):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        if self.horizontal_flip_probability > 0:
            image_array, box_corners = self.horizontal_flip(image_array,
                                                            box_corners)

        if self.vertical_flip_probability > 0:
            image_array, box_corners = self.vertical_flip(image_array,
                                                          box_corners)
        return image_array, box_corners

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)

    def flow(self, mode='train'):
        while True:
            if mode == 'train':
                shuffle(self.train_keys)
                keys = self.train_keys
            elif mode == 'val' or mode == 'demo':
                shuffle(self.validation_keys)
                keys = self.validation_keys
            else:
                raise Exception('invalid mode: %s' % mode)

            inputs = []
            targets = []
            for key in keys:
                image_path = self.path_prefix + key
                image_array = imread(image_path)
                image_array = imresize(image_array, self.image_size)

                num_image_channels = len(image_array.shape)
                if num_image_channels != 3:
                    continue

                ground_truth = self.ground_truth_data[key]

                image_array = image_array.astype('float32')
                if mode == 'train' or mode == 'demo':
                    image_array = self.transform(image_array)[0]

                if self.grayscale:
                    image_array = cv2.cvtColor(image_array.astype('uint8'),
                                               cv2.COLOR_RGB2GRAY).astype('float32')
                    image_array = np.expand_dims(image_array, -1)

                inputs.append(image_array)
                targets.append(ground_truth)
                if len(targets) == self.batch_size:
                    inputs = np.asarray(inputs)
                    targets = np.asarray(targets)

                    gender = np_utils.to_categorical(targets[:, 0], 2)
                    # Quantizing the age

                    age_bins = np.linspace(0, 100, 21)
                    age_step = np.digitize(targets[:, 1], age_bins)
                    age_quantized = np_utils.to_categorical(age_step, 21)

                    if mode == 'train' or mode == 'val':
                        inputs = self.preprocess_images(inputs)
                        yield self._wrap_in_dictionary(inputs, gender, age_quantized)
                    if mode == 'demo':
                        yield self._wrap_in_dictionary(inputs, gender, age_quantized)
                    inputs = []
                    targets = []

    def _wrap_in_dictionary(self, image_array, gender, age_quantized):
        return [
            {'the_input': image_array},
            {'gender': gender, 'age': age_quantized}
        ]
