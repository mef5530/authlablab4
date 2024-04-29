import cv2
import numpy as np
import tensorflow as tf

from training.preprocess import load_img
import deps.augment as augment

def pad(input, ksize, mode, constant_values):
    """
    Function modified from Tensor Flow IO Experimental Repo
    """

    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    mode = "CONSTANT" if mode is None else mode
    constant_values = (
        tf.zeros([], dtype=input.dtype)
        if constant_values is None
        else tf.convert_to_tensor(constant_values, dtype=input.dtype)
    )

    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC")

    height, width = ksize[0], ksize[1]
    top = (height - 1) // 2
    bottom = height - 1 - top
    left = (width - 1) // 2
    right = width - 1 - left
    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    return tf.pad(input, paddings, mode=mode, constant_values=constant_values)

def gabor(
    input,
    freq,
    sigma=None,
    theta=0,
    nstds=3,
    offset=0,
    mode=None,
    constant_values=None,
    name=None,
    max_kernel_size=6
):
    """
    Function modified from Tensor Flow IO Experimental Repo
    """

    """
    Apply Gabor filter to image.

    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      freq: A float Tensor. Spatial frequency of the harmonic function.
        Specified in pixels.
      sigma: A scalar or 1-D `[sx, sy]` Tensor. Standard deviation in
        in x- and y-directions. These directions apply to the kernel
        before rotation. If theta = pi/2, then the kernel is rotated
        90 degrees so that sigma_x controls the vertical direction.
        If scalar, then `sigma` will be broadcasted to 1-D `[sx, sy]`.
      nstd: A scalar Tensor. The linear size of the kernel is nstds
        standard deviations, 3 by default.
      offset: A scalar Tensor. Phase offset of harmonic function in radians.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).

    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """
    input = tf.convert_to_tensor(input)

    dtype = tf.complex128

    freq = tf.cast(freq, dtype.real_dtype)
    if sigma is None:
        # See http://www.cs.rug.nl/~imaging/simplecell.html
        b = 1  # bandwidth
        sigma = (
            tf.cast(
                1.0
                / np.pi
                * np.sqrt(np.log(2) / 2.0)
                * (2.0 ** b + 1)
                / (2.0 ** b - 1),
                dtype.real_dtype,
            )
            / freq
        )
    sigma = tf.broadcast_to(sigma, [2])
    sigma_x, sigma_y = sigma[0], sigma[1]
    theta = tf.cast(theta, dtype.real_dtype)
    nstds = tf.cast(nstds, dtype.real_dtype)
    offset = tf.cast(offset, dtype.real_dtype)

    x0 = tf.math.ceil(
        tf.math.maximum(
            tf.math.abs(nstds * sigma_x * tf.math.cos(theta)),
            tf.math.abs(nstds * sigma_y * tf.math.sin(theta)),
            tf.cast(1, dtype.real_dtype),
        )
    )
    y0 = tf.math.ceil(
        tf.math.maximum(
            tf.math.abs(nstds * sigma_y * tf.math.cos(theta)),
            tf.math.abs(nstds * sigma_x * tf.math.sin(theta)),
            tf.cast(1, dtype.real_dtype),
        )
    )

    x0 = tf.minimum(x0, max_kernel_size)
    y0 = tf.minimum(y0, max_kernel_size)

    y, x = tf.meshgrid(tf.range(-y0, y0 + 1), tf.range(-x0, x0 + 1))
    y, x = tf.transpose(y), tf.transpose(x)

    rotx = y * tf.math.sin(theta) + x * tf.math.cos(theta)
    roty = y * tf.math.cos(theta) - x * tf.math.sin(theta)

    g = tf.math.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g = g / (2 * np.pi * sigma_x * sigma_y)
    g = tf.cast(g, dtype) * tf.exp(
        tf.cast(1j, dtype) * tf.cast(2 * np.pi * freq * rotx + offset, dtype)
    )

    ksize = tf.shape(g)


    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)

    real = tf.nn.depthwise_conv2d(
        input, tf.cast(tf.math.real(g), input.dtype), [1, 1, 1, 1], padding="VALID"
    )

    imag = tf.nn.depthwise_conv2d(
        input, tf.cast(tf.math.imag(g), input.dtype), [1, 1, 1, 1], padding="VALID"
    )


    return tf.complex(real, imag)

def histogram_equalization(img):
    img_flat = tf.reshape(img, [-1])
    
    histogram = tf.histogram_fixed_width(img_flat, [0, 255], nbins=256)
    
    cdf = tf.cumsum(histogram)
    cdf_min = tf.reduce_min(tf.gather(cdf, tf.where(cdf > 0)))  # Ensure cdf_min is extracted from cdf values

    cdf = tf.cast(cdf, tf.int32)
    cdf_min = tf.cast(cdf_min, tf.int32)

    img_shape = tf.shape(img)
    cdf_scaled = (cdf - cdf_min) / (img_shape[0] * img_shape[1] - cdf_min) * 255
    cdf_scaled = tf.cast(cdf_scaled, tf.int32)

    img_eq = tf.gather(cdf_scaled, tf.cast(img_flat, tf.int32))
    img_eq = tf.reshape(img_eq, img_shape)
    
    return img_eq

def gaussian_blur(img, kernel_size=7, sigma=1.5):
    """
    Function modified from https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
    """

    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')

def add_random_rotation(img):
    degrees = tf.random.uniform([], minval=-5, maxval=5, dtype=tf.float32)
    return augment.rotate(img, degrees)

def add_random_translation(image, img_size):
    tx = tf.random.uniform([], minval=-0.1 * img_size, maxval=0.1 * img_size, dtype=tf.float32)
    ty = tf.random.uniform([], minval=-0.1 * img_size, maxval=0.1 * img_size, dtype=tf.float32)
    return augment.translate(image, [tx, ty], fill_value=128.0, fill_mode='constant', interpolation='nearest')

def add_random_contrast(image):
    factor = tf.random.uniform([], minval=1.2, maxval=1.5, dtype=tf.float32)
    return augment.contrast(image, factor)

def add_random_brightness(image):
    factor = tf.random.uniform([], minval=0.8, maxval=1.2, dtype=tf.float32)
    return augment.brightness(image, factor)

def preprocess_and_augment_image(image_path, img_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])

    img = tf.expand_dims(img, axis=0)

    img = add_random_rotation(img)
    img = add_random_translation(img, 400)
    img = add_random_brightness(img)
    img = add_random_brightness(img)

    img = gaussian_blur(img, kernel_size=5, sigma=1)

    freq = 0.27
    theta_values = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    gabor_imgs = [gabor(img, freq=freq, theta=theta, nstds=2, mode='REFLECT') for theta in theta_values]
    gabor_imgs = [tf.math.real(gabor(img, freq=freq, theta=theta)) for theta in theta_values]
    img = tf.reduce_max(gabor_imgs, axis=0)

    img = tf.cast(img, tf.int32)
    img = tf.squeeze(img, axis=[0])
    img = tf.clip_by_value(img, 0, 255)

    img = histogram_equalization(img)

    img = tf.cast(img, tf.uint8)

    img_np = img.numpy()
    thresh, img_otsu = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = tf.convert_to_tensor(img_otsu, dtype=tf.uint8)

    img = tf.expand_dims(img, axis=2)
    return img


def create_nd04_sim_pairs():
    people = load_img()

    img1 = []
    img2 = []
    labels = []

    #similar pairs
    for person in people:
        img1.append(person[0])
        img2.append(person[1])
        labels.append(1)
    
    return img1, img2, labels

pairs = create_nd04_sim_pairs()
img1_fps, img2_fps, labels = pairs

for i in range(30000, 100000):
    print(f'IMG {str(i)}, using {int(i%len(img1_fps))}')
    if i%len(img1_fps) < 1750:
        tf.io.write_file(f'nd04-giga\\ts\\img1\\{i}.png', tf.io.encode_png(preprocess_and_augment_image(img1_fps[i%len(img1_fps)], 400)))
        tf.io.write_file(f'nd04-giga\\ts\\img2\\{i}.png', tf.io.encode_png(preprocess_and_augment_image(img2_fps[i%len(img2_fps)], 400)))
    else:
        tf.io.write_file(f'nd04-giga\\vs\\img1\\{i}.png', tf.io.encode_png(preprocess_and_augment_image(img1_fps[i%len(img1_fps)], 400)))
        tf.io.write_file(f'nd04-giga\\vs\\img2\\{i}.png', tf.io.encode_png(preprocess_and_augment_image(img2_fps[i%len(img2_fps)], 400)))