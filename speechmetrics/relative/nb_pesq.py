from .. import Metric


class NBPESQ(Metric):
    def __init__(self, window, hop=None):
        super(NBPESQ, self).__init__(name='NBPESQ', window=window, hop=hop)
        self.mono = True
        self.fixed_rate = 16000

    def test_window(self, audios, rate):
        # Use pypesq directly since it provides same function name as pesq
        import pypesq
        if len(audios) != 2:
            raise ValueError('NB_PESQ needs a reference and a test signals.')
        return {'nb_pesq': pypesq.pesq(audios[1], audios[0], rate)}


def load(window, hop=None):
    return NBPESQ(window, hop)
