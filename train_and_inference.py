import tensorflow as tf
keras = tf.keras
import soundfile
from audio_model import build_model, Custom_Loss_Fn


def download_dataset():
    """
    This function downloads the dataset and extracts from the zipfiles.
    """
    import wget
    import zipfile

    wget.download("https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip")
    wget.download("https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip")

    with zipfile.ZipFile("noisy_testset_wav.zip", "r") as z_f:
        z_f.extractall("./")
    with zipfile.ZipFile("clean_testset_wav.zip", "r") as z_f:
        z_f.extractall("./")


def preprocess_dataset(noisy_set, clean_set, batch_size, sequence_length):
    """
    This function pairs the inputs and labels into a dataset and processes it.
    In the preprocess step, the .wav file is decoded and the extra dimension at the end is removed.
    """
    def preprocess(noisy_file, clean_file):
        noisy_audio = tf.audio.decode_wav(tf.io.read_file(noisy_file), 1, sequence_length).audio
        noisy_audio = tf.squeeze(noisy_audio, axis=-1)
        clean_audio = tf.audio.decode_wav(tf.io.read_file(clean_file), 1, sequence_length).audio
        clean_audio = tf.squeeze(clean_audio, axis=-1)
        return noisy_audio, clean_audio

    # pairs the inputs and labels together
    dataset = tf.data.Dataset.from_tensor_slices((noisy_set, clean_set))

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size)

    return dataset


def get_training_data(num_train_files, batch_size, sequence_length):
    """
    This function imports the dataset and splits it into training and testing datasets.
    The dataset is split without shuffling so that the files in the test dataset 
    are known and could be used for inference later.
    """

    data_noisy = tf.io.gfile.glob("dataset/noisy_testset_wav/*.wav")
    data_clean = tf.io.gfile.glob("dataset/clean_testset_wav/*.wav")
    
    train_noisy = data_noisy[ : num_train_files] 
    train_clean = data_clean[ : num_train_files]
    test_noisy = data_noisy[num_train_files : ] 
    test_clean = data_clean[num_train_files : ]

    train_data = preprocess_dataset(train_noisy, train_clean, batch_size, sequence_length)
    test_data = preprocess_dataset(test_noisy, test_clean, batch_size, sequence_length)

    return train_data, test_data



def run_training():
    """
    This script trains the model
    """

    batch_size = 32
    sequence_length = 96_000
    intermediate_dim = 512
    num_train_files = 700
    train_data, test_data = get_training_data(num_train_files, batch_size, sequence_length)

    model = build_model(sequence_length, intermediate_dim)
    model.compile(loss=Custom_Loss_Fn(), optimizer=keras.optimizers.Adam())
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_data, validation_data=test_data, epochs=150, callbacks=[callback])
    model.save_weights("audio_weights_1/checkpoint")



def run_inference():
    """
    This function builds the model with the trained weights, takes a noisy audio file as input,
    predicts a clean audio, and saves it as a .wav file.
    """
    sequence_length = 96_000
    intermediate_dim = 512
    new_model = build_model(sequence_length, intermediate_dim, training=False)
    new_model.load_weights("audio_weights/checkpoint").expect_partial()

    noisy_file = tf.io.read_file("samples/Noisy_3_(p257_430).wav")

    # This section predicts only 1 time step (2 seconds)
    noisy_audio, sampling_rate = tf.audio.decode_wav(noisy_file, 1, sequence_length)
    noisy_audio = tf.squeeze(noisy_audio, axis=-1)[tf.newaxis, ...]
    prediction = new_model.predict(noisy_audio)[0]
    soundfile.write("Prediction_3.wav", prediction, sampling_rate.numpy()) 


    # This section predicts 2 time steps of noisy data and concatenates the clips (4 seconds).
    num_time_steps = 2
    noisy_audio, sampling_rate = tf.audio.decode_wav(noisy_file, 1, sequence_length * num_time_steps)
    noisy_audio = tf.squeeze(noisy_audio, axis=-1)

    prediction_list = []
    for i in range(num_time_steps):
        part = noisy_audio[ i * sequence_length : (i + 1) * sequence_length][tf.newaxis, ...]
        prediction_list.append(new_model.predict(part)[0])

    prediction = tf.concat(prediction_list, axis=0)
    soundfile.write("Prediction_3_(2_timesteps).wav", prediction, sampling_rate.numpy()) 




