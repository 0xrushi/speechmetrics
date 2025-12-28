def load(window, hop=None):
    """
    Backwards-compatible `mosnet` metric implemented via DNSMOS (PyTorch).

    The original TensorFlow MOSNet implementation was removed to avoid hard
    TensorFlow/ABI constraints; install `dnsmos` to enable this metric.
    """
    try:
        import dnsmos  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The `mosnet` metric now requires the optional dependency `dnsmos` "
            "(PyTorch). Install it with: `pip install dnsmos`."
        ) from e

    return _DNSMOSAsMOSNet(window, hop)


from ... import Metric


class _DNSMOSAsMOSNet(Metric):
    def __init__(self, window, hop=None, device=None):
        super().__init__(name="DNSMOS", window=window, hop=hop)

        self.fixed_rate = 16000
        self.mono = True
        self.absolute = True

        self._device = device
        self._model = None

    def _default_device(self):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            from dnsmos import DNSMOS
        except ImportError as e:
            raise ImportError(
                "The `mosnet` metric now requires `dnsmos`. "
                "Install it with: `pip install dnsmos`."
            ) from e

        device = self._device or self._default_device()
        self._model = DNSMOS(device=device)
        return self._model

    def test_window(self, audios, rate):
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        model = self._get_model()
        audio = np.asarray(audios[0], dtype=np.float32)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            sf.write(tmp_path, audio, rate)
            result = model(tmp_path)
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        ovrl = _pick_score(result, keys=("OVRL", "ovrl", "overall", "MOS", "mos"))
        sig = _pick_score(result, keys=("SIG", "sig"))
        bak = _pick_score(result, keys=("BAK", "bak"))

        out = {}
        if ovrl is not None:
            out["mosnet"] = float(ovrl)
            out["dnsmos_ovrl"] = float(ovrl)
        if sig is not None:
            out["dnsmos_sig"] = float(sig)
        if bak is not None:
            out["dnsmos_bak"] = float(bak)
        return out


def _pick_score(result, keys):
    if isinstance(result, dict):
        for key in keys:
            if key in result:
                return result[key]
    return None
