# Audio Noise Cancelling using Transformers

* Trained on Tensorflow 2.15.0 and Keras-NLP 0.12.1

## Contents
* [Repository Files](#repository-files)
* [Model](#model)
* [Dataset](#dataset)
* [Inference](#inference)

## Repository Files

* audio_model.py - This file contains the model and the loss function.

* train_and_inference.py - This file contains the preprocessing functions and the scripts for training, inference, and downloading the dataset.

* samples - This folder contains three noisy audio files and their respectively predicted audio files.

* audio_weights - This folder contains the trained weights of the model.

## Model

This model enhances speech (audio) by removing background noise. The design of the model is based on the DTLN model proposed in the paper "Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression" by Nils L. Westhausen and Bernd T. Meyer.

Here, the model uses several modifications to the original DTLN paper. The relations of this model to the DTLN model are described below:

* The DTLN model uses LSTM layers in its "separation cores". This model uses Transformer Decoder layers with causal masks instead of LSTM layers.

* The DTLN model uses two normilization layers which have been omitted in this model.

* Within the DTLN model, there are 4 consecutive steps: conv1D, normalization, separation, and multiplication. After omitting the normalization layer, this model repeats the three remaining steps (conv1D, separation, and multiplication) three times with convolution filters of size 256, 128, and 256. With the layers before and after, the number of filters have the following U-shaped design:\
num_filters = 512 -> 256 -> 128 -> 256 -> 512

* Here, the DTLN paper's signal-to-noise ratio Loss metric is used.

* This model uses parameters different from the DTLN model for several layers.

Furthermore, this model applies a PReLU activation layer following each of the convolution layers and all of the convolution layers use a causal padding.

Reference to the DTLN paper:

```BibTex
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
```

## Dataset

The model is trained on a dataset by Cassia Valentini-Botinhao. It contains pairs of noisy and clean audio files. Their test dataset was used to train this model rather than their training dataset since their test dataset is smaller. It contains 825 pairs of noisy and clean audio files. 

The model was trained on the first 700 pairs of audio files and remaining pairs were used for testing and inference. The dataset was split without shuffling so that the files in the test dataset are known and could be used for inference later.

Reference to the dataset:

```BibTex
Valentini-Botinhao, Cassia. (2017). Noisy speech database for training speech
enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh. 
School of Informatics. Centre for Speech Technology Research (CSTR). 
https://doi.org/10.7488/ds/2117.
```
---

## Inference

The model was trained on noisy audio files decoded at a sampling rate of 48,000 samples per second. The model can denoise 96,000 samples, i.e. 2 seconds of a audio file, at a time. Longer audio files can be denoised by slicing the audio files into 2 second clips and concatenating the results. **Unfortunately, there is noisy static at the points where the audio clips are concatenated.**

Predictions on three noisy files from the test dataset are provided in the samples folder. For the file Noisy_3_(p257_430).wav, predictions for 1 timestep and for 2 timesteps are provided. Notice the noisy static at the 2 second mark in the 2 timesteps prediction file Prediction_3_(2_timesteps).

This model's performance could potentially be improved by:

* using a larger dataset for training

* increasing the number of heads in the transformer layers

* increasing the intermediate dimensions of the transformer layers