"""Uniform predict(seq) -> (pdb_string, info) interface for all methods."""
from __future__ import annotations

import torch

from .esmfold_utils import fold_from_features, lm_features, load_esmfold, output_to_pdb_strings, sequences_to_batch


class ESMFoldPredictor:
    """Original ESMFold (ESM2-3B + folding trunk)."""

    name = "esmfold"

    def __init__(self, device: str, chunk_size: int | None = 64, num_recycles: int | None = None,
                 trunk_dtype: str = "fp32", quant_lm: str = "none", quant_trunk: str = "none"):
        self.device = device
        self.num_recycles = num_recycles
        self.model = load_esmfold(device, chunk_size=chunk_size, load_lm=True, trunk_dtype=trunk_dtype,
                                  quant_lm=quant_lm, quant_trunk=quant_trunk)

    def features(self, seq: str):
        aa, mask, pos = sequences_to_batch([seq], self.device)
        return lm_features(self.model, aa, mask), aa, mask, pos

    @torch.no_grad()
    def predict(self, seq: str, residue_numbers: list[int] | None = None):
        s_lm, aa, mask, pos = self.features(seq)
        out = fold_from_features(self.model, s_lm, aa, mask, pos, num_recycles=self.num_recycles)
        pdb = output_to_pdb_strings(out, [residue_numbers] if residue_numbers else None)[0]
        return pdb, {"plddt": float(out["plddt"][0, :, 1].mean()), "ptm": float(out["ptm"].reshape(-1)[0])}


class QwenFoldPredictor(ESMFoldPredictor):
    """Qwen encoder (+LoRA, adapter) feeding the frozen ESMFold trunk; ESM2 is not loaded."""

    name = "qwenfold"

    def __init__(self, ckpt: str, device: str, chunk_size: int | None = 64, num_recycles: int | None = None,
                 trunk_dtype: str = "fp32", quant_lm: str = "none", quant_trunk: str = "none"):
        from .esm_encoder import load_encoder  # Qwen or small-ESM2 checkpoints
        self.device = device
        self.num_recycles = num_recycles
        self.model = load_esmfold(device, chunk_size=chunk_size, load_lm=False, trunk_dtype=trunk_dtype,
                                  quant_trunk=quant_trunk)
        enc = load_encoder(ckpt)
        if quant_lm == "int8":  # merge LoRA into the base weights, then int8 the backbone
            from .quant import quantize_linears_int8
            if hasattr(enc.backbone, "merge_and_unload"):
                enc.backbone = enc.backbone.merge_and_unload()
            quantize_linears_int8(enc.backbone)
        self.encoder = enc.to(device).eval()

    @torch.no_grad()
    def features(self, seq: str):
        aa, mask, pos = sequences_to_batch([seq], self.device)
        s_lm, _ = self.encoder([seq])
        return s_lm, aa, mask, pos


def count_parameters(predictor) -> dict:
    m = predictor.model
    from .quant import param_bytes
    trunk = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("esm."))
    out = {"folding_params_M": trunk / 1e6}
    if getattr(m, "esm", None) is not None:
        out["lm_params_M"] = sum(p.numel() for p in m.esm.parameters()) / 1e6
    if hasattr(predictor, "encoder"):
        out["lm_params_M"] = sum(p.numel() for p in predictor.encoder.parameters()) / 1e6
        out["trainable_params_M"] = sum(v.numel() for v in predictor.encoder.trainable_state_dict().values()) / 1e6
        out["lm_weight_gb"] = param_bytes(predictor.encoder) / 1024**3
    elif getattr(m, "esm", None) is not None:
        out["lm_weight_gb"] = param_bytes(m.esm) / 1024**3
    out["fold_weight_gb"] = sum(param_bytes(c) for n, c in m.named_children() if n != "esm") / 1024**3
    return out


class TrunkOnlyPredictor(ESMFoldPredictor):
    """Lower bound: the frozen ESMFold trunk with the language-model input set to zero
    (only the amino-acid embedding reaches the trunk).  No language model is loaded."""

    name = "trunkonly"

    def __init__(self, device: str, chunk_size: int | None = 64, num_recycles: int | None = None,
                 trunk_dtype: str = "fp32", quant_lm: str = "none", quant_trunk: str = "none"):
        self.device = device
        self.num_recycles = num_recycles
        self.model = load_esmfold(device, chunk_size=chunk_size, load_lm=False, trunk_dtype=trunk_dtype,
                                  quant_trunk=quant_trunk)

    @torch.no_grad()
    def features(self, seq: str):
        aa, mask, pos = sequences_to_batch([seq], self.device)
        s_lm = torch.zeros(1, len(seq), self.model.esm_s_mlp[-1].out_features, device=self.device)
        return s_lm, aa, mask, pos
