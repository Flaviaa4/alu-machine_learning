#!/usr/bin/env python3
"""Defines a function that creates a variational autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder

    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden
        layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims: integer containing the dimensions of the latent space
        representation

    Returns: encoder, decoder, auto
        encoder is the encoder model, which outputs the latent
            representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    """
    def sampling(args):
        """Reparameterization trick to sample from N(mu, sigma)"""
        mu, log_var = args
        batch = K.shape(mu)[0]
        dims = K.shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dims))
        return mu + K.exp(log_var / 2) * epsilon

    encoder_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder_inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    mu = keras.layers.Dense(latent_dims, activation=None)(encoded)
    log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mu, log_var])
    encoder = keras.Model(encoder_inputs, [z, mu, log_var])

    decoder_inputs = keras.Input(shape=(latent_dims,))
    decoded = decoder_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_inputs, decoded)

    z, mu, log_var = encoder(encoder_inputs)
    auto_outputs = decoder(z)
    auto = keras.Model(encoder_inputs, auto_outputs)

    def vae_loss(inputs, outputs):
        """Computes the VAE loss (reconstruction + KL divergence)"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, outputs) * input_dims
        kl_loss = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
