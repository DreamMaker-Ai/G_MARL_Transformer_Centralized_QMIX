import numpy as np
import tensorflow as tf

from config import Config


class HyperNetwork(tf.keras.models.Model):
    """
    Model: "hyper_network"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense (Dense)               multiple                  2954240

     lambda (Lambda)             multiple                  0

     reshape (Reshape)           multiple                  0

     dense_1 (Dense)             multiple                  147712

     dense_2 (Dense)             multiple                  147712

     lambda_1 (Lambda)           multiple                  0

     dense_3 (Dense)             multiple                  147712

     dense_4 (Dense)             multiple                  257

    =================================================================
    Total params: 3,397,633
    Trainable params: 3,397,633
    Non-trainable params: 0
    _________________________________________________________________
    """
    def __init__(self, hidden_dim, num_agents):
        super(HyperNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.w1 = tf.keras.layers.Dense(units=self.hidden_dim * self.num_agents)
        self.abs_w1 = tf.keras.layers.Lambda(lambda x: tf.abs(x))
        self.reshape_w1 = tf.keras.layers.Reshape(target_shape=(self.num_agents, self.hidden_dim))

        self.b1 = tf.keras.layers.Dense(units=self.hidden_dim)

        self.w2 = tf.keras.layers.Dense(units=self.hidden_dim)
        self.abs_w2 = tf.keras.layers.Lambda(lambda x: tf.abs(x))

        self.b2_1 = tf.keras.layers.Dense(units=self.hidden_dim, activation='relu')
        self.b2_2 = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, x):
        """ x: feature, (b, 576)
            n: num_agents
            dm: hidden=dim
        """

        w1 = self.reshape_w1(self.abs_w1(self.w1(x)))  # (b,n,dm)

        b1 = self.b1(x)  # (b,dm)

        w2 = self.abs_w2(self.w2(x))  # (b, dm)

        b2 = self.b2_2(self.b2_1(x))  # (b,1)

        return [w1, b1, w2, b2]


class MixingNetwork(tf.keras.models.Model):
    """
    Model: "mixing_network"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     activation (Activation)     multiple                  0

     reshape_1 (Reshape)         multiple                  0

    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """
    def __init__(self, hidden_dim, num_agents):
        super(MixingNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.elu = tf.keras.layers.Activation("elu")
        self.reshape = tf.keras.layers.Reshape(target_shape=(1,))

    def call(self, w, q):
        """ q: (b,n) """

        w1 = w[0]  # (b,n,dm)
        b1 = w[1]  # (b, dm)
        w2 = w[2]  # (b, dm)
        b2 = w[3]  # (b, 1)

        y1 = tf.einsum('bij, bi -> bj', w1, q)  # (b,dm)
        y1 = y1 + b1  # (b,dm)

        y2 = self.elu(y1)  # (b,dm)

        y3 = tf.einsum('bi, bi -> b', w2, y2)  # (b,)
        y3 = self.reshape(y3)  # (b,1)

        y3 = y3 + b2

        return y3


class GlobalCNNModel(tf.keras.models.Model):
    """
    Model: "global_cnn_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     conv2d (Conv2D)             multiple                  576

     conv2d_1 (Conv2D)           multiple                  36928

     conv2d_2 (Conv2D)           multiple                  36928

     conv2d_3 (Conv2D)           multiple                  36928

     flatten (Flatten)           multiple                  0

    =================================================================
    Total params: 111,360
    Trainable params: 111,360
    Non-trainable params: 0
    _________________________________________________________________
    """
    def __init__(self, **kwargs):
        super(GlobalCNNModel, self).__init__(**kwargs)

        self.conv0 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=1,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv1 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv2 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.conv3 = \
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=1,
                activation='relu',
                kernel_initializer='Orthogonal'
            )

        self.flatten1 = tf.keras.layers.Flatten()

    @tf.function
    def call(self, inputs):
        # inputs: (b,g,g,ch*n_frames)=(32,15,15,8)

        h = self.conv0(inputs)  # (32,15,15,64)
        h = self.conv1(h)  # (32,7,7,64)
        h = self.conv2(h)  # (32,5,5,64)
        h = self.conv3(h)  # (32,3,3,64)

        feature = self.flatten1(h)  # (32,576)

        return feature


class HyperMixingNetwork(tf.keras.models.Model):
    """
    Model: "hyper_mixing_network"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     global_cnn_model_1 (GlobalC  multiple                 111360
     NNModel)

     hyper_network_1 (HyperNetwo  multiple                 3397633
     rk)

     mixing_network_1 (MixingNet  multiple                 0
     work)

    =================================================================
    Total params: 3,508,993
    Trainable params: 3,508,993
    Non-trainable params: 0
    _________________________________________________________________
    """
    def __init__(self, hidden_dim, num_agents, **kwargs):
        super(HyperMixingNetwork, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.cnn = GlobalCNNModel()
        self.hyper = HyperNetwork(hidden_dim, num_agents)
        self.mixing = MixingNetwork(hidden_dim, num_agents)

    def call(self, x, q):
        """
        :param x: global state, (b,g,g,ch*hyper_mixing_n_frames)
        :param q: Qi, (b,n)
        :return: Q_tot, (b,1)
        """
        feature = self.cnn(x)  # (b,576)

        w = self.hyper(feature)  # weights of mixing network
        # w[0]:(b,n,hidden_dim), w[1]:(b,hidden_dim), w[2]:(b,hidden_dim), w[3]:(b,1)

        q_tot = self.mixing(w, q)  # (b,1)

        return q_tot


def main():
    config = Config()

    """ check conv_network """
    states = np.random.rand(config.batch_size, config.grid_size, config.grid_size,
                            config.hyper_mixing_obs_channels * config.hyper_mixing_n_frames)
    cnn = GlobalCNNModel()

    feature = cnn(states)  # (b, hidden_dim)

    """ check hyper_network """
    hyper = HyperNetwork(config.hyper_mixing_hidden, config.max_num_red_agents)
    [w1, b1, w2, b2] = hyper(feature)
    print(w1.shape)  # (32,20,256)
    print(b1.shape)  # (32,256)
    print(w2.shape)  # (32,256)
    print(b2.shape)  # (32,1)

    """ check mixing_network """
    weights = [w1, b1, w2, b2]
    q = np.random.rand(config.batch_size, config.max_num_red_agents)  # (32,20)

    mixing = MixingNetwork(config.hyper_mixing_hidden, config.max_num_red_agents)
    wq = mixing(weights, q)
    print(wq.shape)  # (32,1)

    """ check hyper_mixing_network """
    hyper_mixing_net = HyperMixingNetwork(config.hyper_mixing_hidden, config.max_num_red_agents)
    q_tot = hyper_mixing_net(states, q)
    print(q_tot.shape)

    hyper_mixing_net.summary()


if __name__ == "__main__":
    main()
