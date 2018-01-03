# coding: utf-8
"""
VAEのサンプルコード
"""
import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L

from sampleXavier import Xavier


class AE(chainer.Chain):
    """AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(AE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h, initialW=Xavier(n_in, n_h))
            self.le2 = L.Linear(n_h, n_latent, initialW=Xavier(n_h, n_latent))
            # decoder
            self.ld1 = L.Linear(n_latent, n_h, initialW=Xavier(n_latent, n_h))
            self.ld2 = L.Linear(n_h, n_in, initialW=Xavier(n_h, n_in))

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2(h1)
        return mu, None

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self):
        def lf(x):
            mu, _ = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            loss = F.bernoulli_nll(x, self.decode(mu, sigmoid=False)) \
                    / (batchsize)
            self.rec_loss = loss
            self.loss = loss
            chainer.report(
                    {'rec_loss': loss, 'loss': loss}, observer=self)
            return self.rec_loss
        return lf
