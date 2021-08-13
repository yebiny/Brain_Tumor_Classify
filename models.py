import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 
                "embed_dim":self.embed_dim,
                "num_heads":self.num_heads} 


def embedding_block(x):
    y = layers.Reshape([64,32*32, 128], )(x)
    y = layers.LayerNormalization(epsilon=1e-6, name="Emb_norm")(y)
    y = layers.Dense(16, name="Emb_channel")(y)
    y = K.permute_dimensions(y, (0,1,3,2))
    y = layers.Dense(32, name="Emb_spatial")(y)
    y = layers.Reshape([64,32*16])(y)
    return y

def mh_encoder_block(x, i=0):
    y = layers.LayerNormalization(epsilon=1e-6, name="MH%i_norm"%i)(x)
    y = layers.Concatenate(name="MH%i_concat"%i)([MultiHeadSelfAttention(64, 8)(y) for _ in range(8)])
    z = layers.Add(name="MH%i_add"%i)([y, x])

    y = layers.LayerNormalization(epsilon=1e-6, name="MLP%i_norm"%i)(z)
    y = layers.Dense(128, activation=tfa.activations.gelu, name="MLP%i_hidden"%i)(y)
    y = layers.Dropout(0.2, name="MLP%i_dropout"%i)(y)
    y = layers.Dense(512, name="MLP%i_dense"%i)(y)
    y = layers.Add(name="MLP%i_add"%i)([y, z])
    return y

def mlp_head(x):
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(128, activation=tfa.activations.gelu)(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(2)(y)
    return y

def build_model(input_shape):
    x = layers.Input(shape=input_shape)
    y = embedding_block(x)
    y = mh_encoder_block(y, 1)
    y = mh_encoder_block(y, 2)
    y = mh_encoder_block(y, 3)
    y = mh_encoder_block(y, 4)
    
    #y = mlp_head(y[:,0])
    y = layers.GlobalAveragePooling1D()(y)
    y = layers.Dense(1, activation='sigmoid')(y)
    model = models.Model(x, y)

    return model

def main():
    input_shape=(64,32,32,128)
    model = build_model(input_shape)
    model.summary()

if __name__=='__main__':
    main()
