from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow import keras


def create_cnn_base(input_shape, conv_layers, dense_layers, dropout: float):
    input = keras.Input(shape=input_shape, name="input")

    x = None
    for index, conv_layer in enumerate(conv_layers):
        count = index + 1
        n_neurons = conv_layer['n_neurons']
        kernel_size = conv_layer['kernel_size']
        l2_reg = conv_layer['l2_regularizer']
        l2_bias_reg = conv_layer['l2_bias_regularizer']

        if (index == 0):
            x = Conv1D(n_neurons,
                       kernel_size=kernel_size,
                       activation='relu',
                       input_shape=input_shape,
                       kernel_regularizer=regularizers.L2(l2=l2_reg),
                       bias_regularizer=regularizers.L2(l2_bias_reg),
                       name=f'conv1d-{count}')(input)
        else:
            x = Conv1D(n_neurons,
                       kernel_size=kernel_size,
                       activation='relu',
                       kernel_regularizer=regularizers.L2(l2=l2_reg),
                       bias_regularizer=regularizers.L2(l2_bias_reg),
                       name=f'conv1d-{count}')(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = Dropout(dropout)(x)
    x = Flatten(name="flatten-1")(x)

    for index, dense_layer in enumerate(dense_layers):
        count = index + 1
        n_neurons = dense_layer['n_neurons']
        l2_reg = dense_layer['l2_regularizer']
        l2_bias_reg = dense_layer['l2_bias_regularizer']

        x = Dense(n_neurons,
                  activation='relu',
                  kernel_regularizer=regularizers.L2(l2=l2_reg),
                  bias_regularizer=regularizers.L2(l2_bias_reg),
                  name=f'dense-{count}')(x)

    output = Dense(1, activation='sigmoid', name="output")(x)

    model = keras.Model(inputs=input, outputs=output)
    return model
