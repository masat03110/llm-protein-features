"""ESMFold split into two halves so the language model can be swapped.

    sequence --[ESM2-3B]--> esm_s (L x 37 x 2560) --[combine + esm_s_mlp]--> s_lm (L x 1024)
    s_lm + aa_embedding --> folding trunk (48 blocks) --> structure module --> 3D

``lm_features`` returns ``s_lm`` (the distillation target for the Qwen encoder),
``fold_from_features`` runs everything after it.  Running the two in sequence
reproduces ``EsmForProteinFolding.forward`` exactly (checked by
``scripts/check_esmfold_split.py``), but unlike the HF implementation we do not
detach ``s_lm`` so a structure loss can back-propagate into the encoder.
"""
from __future__ import annotations

import torch
from transformers import EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import (
    EsmForProteinFoldingOutput,
    categorical_lddt,
    compute_predicted_aligned_error,
    compute_tm,
)
from transformers.models.esm.openfold_utils import make_atom14_masks, residue_constants

ESMFOLD_PATH = "models/esmfold_v1"  # local copy of facebook/esmfold_v1 (scripts/fetch_hf_repo.py)


def load_esmfold(device: str, path: str = ESMFOLD_PATH, chunk_size: int | None = 64,
                 load_lm: bool = True, trunk_dtype: str = "fp32", quant_lm: str = "none",
                 quant_trunk: str = "none") -> EsmForProteinFolding:
    """trunk_dtype: "fp32" (default, as in HF) or "bf16" (everything except ESM2 cast to bfloat16:
    halves the 693M-parameter trunk from 2.8 GB to 1.4 GB; accuracy must be checked per experiment).
    quant_lm / quant_trunk = "int8": weight-only int8 for ESM2-3B / the folding trunk (qwenfold.quant)."""
    from .quant import quantize_linears_int8
    model = EsmForProteinFolding.from_pretrained(path, dtype=torch.float32)
    esm = model.esm
    model.esm = None
    if trunk_dtype == "bf16":
        model = model.to(torch.bfloat16)
    elif trunk_dtype != "fp32":
        raise ValueError(trunk_dtype)
    if quant_trunk == "int8":
        quantize_linears_int8(model.trunk)
    if load_lm:
        model.esm = esm.half()
        if quant_lm == "int8":
            quantize_linears_int8(model.esm)
    else:
        # The trunk alone: drop the 3B-parameter ESM2 to save ~6 GB.
        del esm
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.trunk.set_chunk_size(chunk_size)
    return model


def sequences_to_batch(seqs: list[str], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AF2-style aatype indices, mask and position ids (right-padded)."""
    L = max(len(s) for s in seqs)
    aa = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx = [residue_constants.restype_order_with_x.get(c, residue_constants.restype_num) for c in s]
        aa[i, : len(s)] = torch.tensor(idx)
        mask[i, : len(s)] = 1
    pos = torch.arange(L).expand(len(seqs), -1).clone()
    return aa.to(device), mask.to(device), pos.to(device)


@torch.no_grad()
def lm_features(model: EsmForProteinFolding, aa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ESM2 representation after layer-combination and esm_s_mlp: (B, L, 1024)."""
    esmaa = model.af2_idx_to_esm_idx(aa, mask)
    esm_s = model.compute_language_model_representations(esmaa).to(model.esm_s_combine.dtype)
    esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    return model.esm_s_mlp(esm_s)


def fold_from_features(model: EsmForProteinFolding, s_lm: torch.Tensor, aa: torch.Tensor,
                       mask: torch.Tensor, position_ids: torch.Tensor,
                       num_recycles: int | None = None) -> EsmForProteinFoldingOutput:
    """Everything in EsmForProteinFolding.forward after the language model."""
    cfg = model.config.esmfold_config
    B, L = aa.shape
    s_s_0 = s_lm.to(model.esm_s_combine.dtype)
    s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)
    if cfg.embed_aa:
        s_s_0 = s_s_0 + model.embedding(aa)
    structure = model.trunk(s_s_0, s_z_0, aa, position_ids, mask, no_recycles=num_recycles)
    keep = ["s_z", "s_s", "frames", "sidechain_frames", "unnormalized_angles", "angles", "positions", "states"]
    structure = {k: v for k, v in structure.items() if k in keep}
    disto = model.distogram_head(structure["s_z"])
    structure["distogram_logits"] = (disto + disto.transpose(1, 2)) / 2
    structure["lm_logits"] = model.lm_head(structure["s_s"])
    structure["aatype"] = aa
    make_atom14_masks(structure)
    for k in ["atom14_atom_exists", "atom37_atom_exists"]:
        structure[k] *= mask.unsqueeze(-1)
    structure["residue_index"] = position_ids
    lddt_head = model.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, model.lddt_bins)
    structure["lddt_head"] = lddt_head
    structure["plddt"] = categorical_lddt(lddt_head[-1], bins=model.lddt_bins)
    ptm_logits = model.ptm_head(structure["s_z"])
    structure["ptm_logits"] = ptm_logits
    structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=model.distogram_bins)
    structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=model.distogram_bins))
    return EsmForProteinFoldingOutput(**structure)


def output_to_pdb_strings(output, residue_numbers: list[list[int]] | None = None) -> list[str]:
    """PDB strings; optionally renumber residues (e.g. CASP numbering)."""
    out = {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
    if residue_numbers is not None:
        rn = torch.zeros_like(out["residue_index"])
        for i, nums in enumerate(residue_numbers):
            rn[i, : len(nums)] = torch.tensor(nums) - 1  # output_to_pdb adds 1
        out["residue_index"] = rn
    out = {k: v.float() if v.is_floating_point() else v for k, v in out.items()}
    return EsmForProteinFolding.output_to_pdb(out)
