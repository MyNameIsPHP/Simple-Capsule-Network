import tensorflow as tf
from tensorflow.keras import layers, models

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape [None, input_num_capsules, input_dim_capsule]"
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

    def call(self, inputs):
        inputs_expanded = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expanded, [1, self.num_capsules, 1, 1])
        inputs_tiled = tf.expand_dims(inputs_tiled, 4)

        # Compute 'u_hat' by dot product of 'W' and 'inputs' [batch_size, num_capsules, input_num_capsules, dim_capsule, 1]
        u_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        # Initialize the logits to zero.
        b_ij = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, self.input_num_capsules, 1, 1])

        # Routing algorithm
        for i in range(self.routings):
            c_ij = tf.nn.softmax(b_ij, axis=1)
            s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=2, keepdims=True)
            v_j = squash(s_j)

            if i < self.routings - 1:
                b_ij += tf.matmul(u_hat, v_j, transpose_a=True)

        return tf.squeeze(v_j, axis=[2, 4])
