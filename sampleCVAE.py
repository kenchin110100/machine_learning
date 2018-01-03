# coding: utf-8
"""
CVAEのサンプルコード
"""
import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L

from sampleXavier import Xavier


class CVAE(chainer.Chain):
    """Conditional Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h, n_label):
        super(CVAE, self).__init__()
        with self.init_scope():
            # encoder
            self.embed_e = L.EmbedID(n_label, n_h, ignore_label=-1, initialW=Xavier(n_label, n_h))
            self.le1 = L.Linear(n_in, n_h, initialW=Xavier(n_in, n_h))
            self.le2_mu = L.Linear(n_h*2, n_latent, initialW=Xavier(n_h*2, n_latent))
            self.le2_ln_var = L.Linear(n_h*2, n_latent, initialW=Xavier(n_h*2, n_latent))
            # decoder
            self.embed_d = L.EmbedID(n_label, n_h, ignore_label=-1, initialW=Xavier(n_label, n_h))
            self.ld1 = L.Linear(n_latent, n_h, initialW=Xavier(n_latent, n_h))
            self.ld2 = L.Linear(n_h*2, n_in, initialW=Xavier(n_h*2, n_in))

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x, y):
        h1 = F.tanh(self.le1(x))
        h2 = F.tanh(self.embed_e(y))
        mu = self.le2_mu(F.concat([h1, h2]))
        ln_var = self.le2_ln_var(F.concat([h1, h2]))  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, y, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = F.tanh(self.embed_d(y))
        h3 = self.ld2(F.concat([h1, h2]))
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x, y):
            mu, ln_var = self.encode(x, y)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, y, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                C * gaussian_kl_divergence(mu, ln_var) / batchsize
            return self.loss
        return lf
