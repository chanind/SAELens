from pathlib import Path

import torch

from sae_lens.sae import SAE
from sae_lens.training.training_sae import TrainingSAE
from tests.helpers import build_sae_cfg


def test_matching_pursuit_sae_initialization():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "matching_pursuit")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # check if the decoder weight norm is 1 by default
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)))


def test_matching_pursuit_sae_forward_pass_works_with_3d_inputs():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "matching_pursuit")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    d_in = sae.cfg.d_in

    x = torch.randn(2, 10, d_in)
    sae_out = sae(x)
    assert sae_out.shape == (2, 10, d_in)


def test_matching_pursuit_sae_save_and_load_works(tmp_path: Path):
    cfg = build_sae_cfg(matching_pursuit_maxk=7, matching_pursuit_threshold=0.3)
    setattr(cfg, "architecture", "matching_pursuit")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    sae.save_model(tmp_path)

    loaded_sae = SAE.load_from_pretrained(str(tmp_path))

    assert loaded_sae.cfg.matching_pursuit_maxk == 7
    assert loaded_sae.cfg.matching_pursuit_threshold == 0.3
    assert loaded_sae.cfg.architecture == "matching_pursuit"
