import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Dropout, concatenate, SpatialDropout2D, Add
from keras.regularizers import l2
from keras.initializers import HeNormal, GlorotUniform

def resnet_identity_block(x, filters, kernel_size=3):
    fx = Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(2e-4))(x)
    fx = BatchNormalization()(fx)
    fx = Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(2e-4))(fx)
    fx = BatchNormalization()(fx)
    out = Add()([x, fx])  # Skip connection
    return out

def build_resnet_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(2e-4))(input)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = resnet_identity_block(x, 64)
    x = resnet_identity_block(x, 64)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(2e-4))(x)
    x = resnet_identity_block(x, 128)
    x = resnet_identity_block(x, 128)
    x = Flatten()(x)
    x = Dense(256, activation='sigmoid', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(1e-3))(x)
    return Model(inputs=input, outputs=x)

def get_towers(input_name, IMAGE_SIZE_D):
    input_shape = (IMAGE_SIZE_D, IMAGE_SIZE_D, 1)

    tower = build_resnet_network(input_shape=input_shape)

    input_a = Input(shape=input_shape, name=f'{input_name}_a')
    input_b = Input(shape=input_shape, name=f'{input_name}_b')

    return tower(input_a), tower(input_b), input_a, input_b

def get_model(IMAGE_SIZE_D):
    processed_img_a, processed_img_b, input_img_a, input_img_b = get_towers('input_img', IMAGE_SIZE_D)

    combined_a = concatenate([processed_img_a])
    combined_b = concatenate([processed_img_b])

    @keras.saving.register_keras_serializable()
    def man_dist(embeddings):
        import tensorflow as tf
        return tf.abs(embeddings[0] - embeddings[1])

    distance = Lambda(man_dist, output_shape=lambda shapes: shapes[0])([combined_a, combined_b])

    normalized_distance = BatchNormalization()(distance)
    outputs = Dense(1, activation='sigmoid')(normalized_distance)

    model = Model(inputs=[input_img_a, input_img_b], outputs=outputs)
    return model