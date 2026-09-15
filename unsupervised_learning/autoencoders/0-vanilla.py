#!/usr/bin/env python3
"""Defines a function that creates a vanilla autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder

    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden
        layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims: integer containing the dimensions of the latent space
        representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    encoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder_inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(encoder_inputs, latent)

    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_inputs, decoded)

    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(encoder_inputs, auto_outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
