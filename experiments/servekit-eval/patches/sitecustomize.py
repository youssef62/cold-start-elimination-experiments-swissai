"""Raise sglang's post-load barrier timeout, which is a module constant.

`dist_barrier_after_load` calls `monitored_barrier(timeout=
UNBALANCED_MODEL_LOADING_TIMEOUT_S)` per TP group. The constant is hardcoded at
480 s, only rank 0 observes it, and a cold 1.5 TB read off Lustre has shown
within-node spreads well past that -- so whenever the slowest rank in a group is
not rank 0, rank 0 raises and kills the whole engine.

Python auto-imports sitecustomize at interpreter startup, before sglang runs, so
this reaches the scheduler subprocesses too; putting the directory on PYTHONPATH
is the whole installation.
"""
import os

try:
    import sglang.srt.model_executor.model_runner_components.load_model_utils as _lm

    _lm.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("UNBALANCED_MODEL_LOADING_TIMEOUT_S", "3600")
    )
except Exception:
    pass


# sglang's fastsafetensors path picks the device as cuda:{pg.rank()}, using the
# GLOBAL rank: on a 4-GPU node, rank 9 asks for cuda:9 and is_gds_supported
# fails with "invalid device ordinal". It is only correct when world size fits
# in one node, which is why the single-node TP4 sweep never hit it.
#
# Rewritten at the boundary rather than by reimplementing sglang's iterator, and
# only when the index is out of range, so a correct caller is left alone. Gated
# by an env var because it is a workaround for an upstream bug, not something
# the other arms should silently inherit.
if os.environ.get("SERVEKIT_FST_DEVICE_FIX"):
    try:
        import torch
        from fastsafetensors.loader import SafeTensorsFileLoader as _FSTLoader

        _fst_init = _FSTLoader.__init__

        def _fst_init_local_device(self, pg, device, *args, **kwargs):
            index = getattr(device, "index", None)
            if getattr(device, "type", None) == "cuda" and index is not None:
                if index >= torch.cuda.device_count():
                    device = torch.device("cuda", torch.cuda.current_device())
            return _fst_init(self, pg, device, *args, **kwargs)

        _FSTLoader.__init__ = _fst_init_local_device
    except Exception:
        pass
