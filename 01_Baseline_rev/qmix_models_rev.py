import numpy as np
import tensorflow as tf

from config import Config


class HyperNetwork(tf.keras.models.Model):
    """
    Model: "hyper_network"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     dense (Dense)               multiple                  65792

     lambda (Lambda)             multiple                  0

     reshape (Reshape)           multiple                  0

     dense_1 (Dense)             multiple                  1310976

     dense_2 (Dense)             multiple                  1310976

     lambda_1 (Lambda)           multiple                  0

     dense_3 (Dense)             multiple                  1310976

     dense_4 (Dense)             multiple                  257

    =================================================================
    Total params: 3,998,977
    Trainable params: 3,998,977
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, hidden_dim, num_agents):
        super(HyperNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.w1 = tf.keras.layers.Dense(units=self.hidden_dim)
        self.abs_w1 = tf.keras.layers.Lambda(lambda x: tf.abs(x))

        self.reshape = tf.keras.layers.Reshape(target_shape=(self.num_agents * self.hidden_dim,))

        self.b1 = tf.keras.layers.Dense(units=self.hidden_dim)

        self.w2 = tf.keras.layers.Dense(units=self.hidden_dim)
        self.abs_w2 = tf.keras.layers.Lambda(lambda x: tf.abs(x))

        self.b2_1 = tf.keras.layers.Dense(units=self.hidden_dim, activation='relu')
        self.b2_2 = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, x):
        """ x: masked_features, (b,n,hidden_dim)=(b,20,256)
            n: num_agents
            dm: hidden_dim
        """

        w1 = self.abs_w1(self.w1(x))  # (b,n,dm)

        y = self.reshape(x)  # (b,n*dm)=(b,20*256)

        b1 = self.b1(y)  # (b,dm)

        w2 = self.abs_w2(self.w2(y))  # (b, dm)

        b2 = self.b2_2(self.b2_1(y))  # (b,1)

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

    def __init__(self):
        super(MixingNetwork, self).__init__()
        self.elu = tf.keras.layers.Activation("elu")
        self.reshape = tf.keras.layers.Reshape(target_shape=(1,))

    def call(self, w, q):
        """ q: (b,n) """

        w1 = w[0]  # (b,n,dm)
        b1 = w[1]  # (b,dm)
        w2 = w[2]  # (b,dm)
        b2 = w[3]  # (b,1)

        y1 = tf.einsum('bij, bi -> bj', w1, q)  # (b,dm)
        y1 = y1 + b1  # (b,dm)

        y2 = self.elu(y1)  # (b,dm)

        y3 = tf.einsum('bi, bi -> b', w2, y2)  # (b,)
        y3 = self.reshape(y3)  # (b,1)

        y3 = y3 + b2

        return y3


class HyperMixingNetwork(tf.keras.models.Model):
    """
    Model: "hyper_mixing_network"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     hyper_network_1 (HyperNetwo  multiple                 3998977
     rk)

     mixing_network_1 (MixingNet  multiple                 0
     work)

    =================================================================
    Total params: 3,998,977
    Trainable params: 3,998,977
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, hidden_dim, num_agents, **kwargs):
        super(HyperMixingNetwork, self).__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.hyper = HyperNetwork(hidden_dim, num_agents)
        self.mixing = MixingNetwork()

    def call(self, x, q):
        """
        :param x: global state=masked_features, (b,n,hidden_dim)
        :param q: Qi, (b,n)
        :return: Q_tot, (b,1)
        """

        w = self.hyper(x)  # weights of mixing network
        # w[0]:(b,n,hidden_dim), w[1]:(b,hidden_dim), w[2]:(b,hidden_dim), w[3]:(b,1)

        q_tot = self.mixing(w, q)  # (b,1)

        return q_tot


def main():
    config = Config()

    masked_features = np.random.rand(
        config.batch_size,
        config.max_num_red_agents,
        config.hidden_dim
    )  # (32,20,256)

    """ check hyper_network """
    hyper = HyperNetwork(config.hyper_mixing_hidden, config.max_num_red_agents)
    [w1, b1, w2, b2] = hyper(masked_features)
    print(w1.shape)  # (32,20,256)
    print(b1.shape)  # (32,256)
    print(w2.shape)  # (32,256)
    print(b2.shape)  # (32,1)

    """ check mixing_network """
    weights = [w1, b1, w2, b2]
    q = np.random.rand(config.batch_size, config.max_num_red_agents)  # (32,20)

    mixing = MixingNetwork()
    wq = mixing(weights, q)
    print(wq.shape)  # (32,1)

    """ check hyper_mixing_network """
    hyper_mixing_net = HyperMixingNetwork(config.hyper_mixing_hidden, config.max_num_red_agents)
    q_tot = hyper_mixing_net(masked_features, q)
    print(q_tot.shape)

    hyper_mixing_net.summary()


if __name__ == "__main__":
    main()
