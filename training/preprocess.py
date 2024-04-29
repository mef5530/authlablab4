import tensorflow as tf
import os
import random
import logging

from training.settings import IMAGE_SIZE_D

logger = logging.getLogger(__name__)

def preprocess_and_augment_image(image_path, img_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)  # Auto-detects the image format
    img = tf.image.resize(img, [img_size, img_size])
   
    img = tf.cast(img, tf.float32) / 255.0

    img.set_shape([img_size, img_size, 1])

    return img

def load_img(folder='sd04\\png_txt\\', unique_person_s=2000):
    people = [[] for _ in range(unique_person_s)]
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith('.png'):
                people[int(fn[1:5])-1].append(os.path.join(root, fn))
    
    return people

def create_pairs():
    people = load_img()

    img1 = []
    img2 = []
    labels = []

    #similar pairs
    for person in people:
        img1.append(person[0])
        img2.append(person[1])
        labels.append(1)

    #dissimilar
    for i in range(len(people)):
        while(True):
            ri1 = random.randint(0, len(people)-1)
            ri2 = random.randint(0, len(people)-1)

            if ri1 != ri2:
                break
        
        img1.append(people[ri1][0])
        img2.append(people[ri2][1])
        labels.append(0)
    
    return img1, img2, labels

def create_synfing_pair(fp1, fp2, pair_len, f_ending='.png'):
    img = []
    img1 = []
    labels = []

    logger.debug('preprocessing: starting to load FPs')

    for root, _, files in os.walk(fp1):
        for fn in files:
            if len(img) > pair_len: break

            if fn.endswith(f_ending):
                img.append(os.path.join(root, fn))
    
    for root, _, files in os.walk(fp2):
        for fn in files:
            if len(img1) > pair_len: break

            if fn.endswith(f_ending):
                img1.append(os.path.join(root, fn))
                labels.append(1)
    
    for i in range(len(img)):
        if i > pair_len:
            break

        while(True):
            ri1 = random.randint(0, len(img)-1)
            ri2 = random.randint(0, len(img)-1)

            if ri1 != ri2:
                break
        
        img.append(img[ri1])
        img1.append(img1[ri2])
        labels.append(0)

    logger.debug('preprocessing: finished loading FPs')

    return img, img1, labels

def process_images(img_1, img_2, label):
    img1 = preprocess_and_augment_image(img_1, img_size=IMAGE_SIZE_D)
    img2 = preprocess_and_augment_image(img_2, img_size=IMAGE_SIZE_D)
    return (img1, img2), label

def create_dataset(img_paths1, img_paths2, labels, shuffle_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths1, img_paths2, labels))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(process_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset.repeat()