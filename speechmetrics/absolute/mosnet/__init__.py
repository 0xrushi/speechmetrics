def load(window, hop=None):
    """
    Backwards-compatible `mosnet` metric implemented via a non-TensorFlow MOS backend.

    The original TensorFlow MOSNet implementation was removed to avoid hard
    TensorFlow/ABI constraints. Install a supported backend to enable this
    metric (see error message for options).
    """
    try:
        from speechmos.dnsmos import run as _speechmos_run  # noqa: F401
        return _SpeechMOSDNSMOSAsMOSNet(window, hop)
    except ImportError:
        pass

    try:
        from dnsmos import DNSMOS  # noqa: F401
        return _DNSMOSAsMOSNet(window, hop)
    except ImportError as e:
        raise ImportError(
            "The `mosnet` metric requires a non-TensorFlow MOS backend, but none "
            "is installed. Try one of:\n"
            "  - `pip install speechmos` (DNSMOS via ONNXRuntime; ships models)\n"
            "  - Provide a compatible `dnsmos` package in your index\n"
            "Or load metrics with `exclude=['mosnet']`."
        ) from e


from ... import Metric


class _SpeechMOSDNSMOSAsMOSNet(Metric):
    def __init__(self, window, hop=None, model_type="dnsmos"):
        super().__init__(name="DNSMOS", window=window, hop=hop)

        self.fixed_rate = 16000
        self.mono = True
        self.absolute = True

        self._model_type = model_type

    def test_window(self, audios, rate):
        import numpy as np

        from speechmos.dnsmos import run as dnsmos_run

        audio = np.asarray(audios[0], dtype=np.float32)
        if audio.size == 0:
            return {"mosnet": np.nan}

        # speechmos requires float audio in [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        result = dnsmos_run(audio, rate, model_type=self._model_type, return_df=True, verbose=False)

        ovrl = result.get("ovrl_mos")
        sig = result.get("sig_mos")
        bak = result.get("bak_mos")
        p808 = result.get("p808_mos")

        out = {}
        if ovrl is not None:
            out["mosnet"] = float(ovrl)
            out["dnsmos_ovrl"] = float(ovrl)
        if sig is not None:
            out["dnsmos_sig"] = float(sig)
        if bak is not None:
            out["dnsmos_bak"] = float(bak)
        if p808 is not None:
            out["dnsmos_p808"] = float(p808)
        return out


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
                "The `mosnet` metric requires a non-TensorFlow MOS backend. "
                "Try `pip install speechmos` (recommended), or provide a "
                "compatible `dnsmos` package in your index."
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
