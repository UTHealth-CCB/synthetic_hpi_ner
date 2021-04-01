import os

from utils.metrics.Metrics import Metrics


class Scalar(Metrics):
    def __init__(self, sess, scalar, name):
        super().__init__()
        self.name = name
        self.sess = sess
        self.scalar = scalar

    def get_score(self):
        return self.sess.run(self.scalar)
    