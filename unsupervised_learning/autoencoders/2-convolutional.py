#!/usr/bin/env python3
"""Defines a function that creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    input_dims: tuple of integers containing the dimensions of the model
        input
    filters: list containing the number of filters for each convolutional
        layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims: tuple of integers containing the dimensions of the latent
        space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    encoder_inputs = keras.Input(shape=input_dims)
    encoded = encoder_inputs
    for f in filters:
        encoded = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(encoded)
        encoded = keras.layers.MaxPooling2D(
            (2, 2), padding='same')(encoded)
    encoder = keras.Model(encoder_inputs, encoded)

    decoder_inputs = keras.Input(shape=latent_dims)
    decoded = decoder_inputs
    reversed_filters = list(reversed(filters))
    for f in reversed_filters[:-1]:
        decoded = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        reversed_filters[-1], (3, 3), padding='valid',
        activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same',
        activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_inputs, decoded)

    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
