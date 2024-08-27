import tensorflow as tf
import keras_nlp
import numpy as np
keras = tf.keras
layers = tf.keras.layers


class Transformer_Layer(layers.Layer):
    """ 
    This is a Transformer Decoder layer that uses a causal mask 
    """
    def __init__(self, intermediate_dim, training):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.training = training

    def build(self, input_shape):
        self.transformer = keras_nlp.layers.TransformerDecoder(intermediate_dim=self.intermediate_dim,
                                                                num_heads=1, dropout=0.1)
        self.sigmoid = keras.activations.sigmoid

    # output shape is equal to the input shape
    def call(self, x):
        mask = self.transformer(x, training=self.training)

        # applies a sigmoid activation to create the mask
        mask = self.sigmoid(mask)
        return mask


class Inverse_Layer(layers.Layer):
    """
    This layer calculates the inverse fourier transform of fft results
    """
    def __init__(self):
        super().__init__()

    def call(self, magnitude, phase):
        magnitude_complex = tf.cast(magnitude, tf.complex64)
        phase_complex = 1j * tf.cast(phase, tf.complex64)

        # combines the complex forms of the magnitude and phase
        stft_complex = tf.math.multiply(magnitude_complex, tf.exp(phase_complex))

        time_frames = tf.signal.irfft(stft_complex)
        return time_frames


class Causal_Conv1D(layers.Layer):
    """
    This layer applies a convolution with causal padding and a PReLU activation
    """
    def __init__(self, filters=16, kernel_size=1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='causal')
        self.activation = layers.PReLU()

    def call(self, input):
        x = self.conv(input)
        x = self.activation(x)
        return x


def build_model(sequence_length=96_000, intermediate_dim=128, frame_steps=256, training=True):

    # input shape = (batch_size, sequence_length)
    input = keras.Input([sequence_length,])

    # time_frames = 374 = n+1 where (frame_step * n + frame_length) < signals.shape[1]
    # frequency_bins = 257 = 2^n + 1 where 2^n + 1 < frame_length
    # stft shape = (batch_size, 374, 257)
    stft = tf.signal.stft(signals=input, frame_length=512, frame_step=frame_steps)

    # magnitude shape = (batch_size, 374, 257)
    magnitude = tf.math.abs(stft)

    # phase shape = (batch_size, 374, 257)
    phase = tf.math.angle(stft)

    eps = np.finfo(np.float32).eps.item()
    log_magnitude = tf.math.log(magnitude + eps)

    # mask shape = (batch_size, 374, 257)
    mask = Transformer_Layer(intermediate_dim, training)(log_magnitude)

    # Applying the mask to the magnitude
    # masked_magnitude shape = (batch_size, 374, 257)
    masked_magnitude = layers.Multiply()([magnitude, mask])

    # time_frames shape = (batch_size, 374, 512)
    time_frames = Inverse_Layer()(masked_magnitude, phase)

    # U-shaped architecture: num_filters = 512 -> 256 -> 128 -> 256 -> 512
    for filters in [256, 128, 256]:

        # time_frames shape = (batch_size, 374, num_filter)
        time_frames = Causal_Conv1D(filters, kernel_size=1)(time_frames)

        # mask shape = (batch_size, 374, num_filter)
        mask = Transformer_Layer(intermediate_dim, training)(time_frames)

        # masked_frames shape = (batch_size, 374, num_filter)
        time_frames = layers.Multiply()([time_frames, mask])

    # time_frames shape = (batch_size, 374, 512)
    time_frames = Causal_Conv1D(filters=512, kernel_size=1)(time_frames)

    # time_space shape = (batch_size, 96_000)
    time_space = tf.signal.overlap_and_add(signal=time_frames, frame_step=frame_steps)

    return keras.Model(input, time_space)


class Custom_Loss_Fn(tf.losses.Loss):
    """
    This loss function uses the Signal To Noise Ratio (SNR).
    This function is designed according to the DTLN paper and its implementation.
    """
    def __init__(self):
        super().__init__()
        self.mse = keras.losses.MSE
    
    def call(self, y_true, y_pred):
        true_square = tf.math.reduce_mean(tf.math.square(y_true), axis=-1)
        signal_to_noise = tf.math.divide_no_nan(true_square, self.mse(y_true, y_pred))

        # loss = - 10 * log(SNR)
        loss = - 10 * tf.experimental.numpy.log10(signal_to_noise)
        return tf.math.reduce_mean(loss)








