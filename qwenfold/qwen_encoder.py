"""General-purpose LLM (Qwen3.5) as a per-residue protein encoder for ESMFold.

    sequence -> Qwen (frozen weights + LoRA) -> learned layer mix -> adapter MLP -> s_lm (L x 1024)

s_lm replaces ESMFold's ESM2-derived representation (see esmfold_utils).

Tokenisation: one residue = one token (" A", " C", ...), so hidden states map
1:1 onto residues.

Context modes (Qwen is causal; ESM2 is bidirectional):
  * "causal": prefix + seq; residue i only sees residues <= i.
  * "echo":   prefix + seq + separator + seq; states of the 2nd copy see the
              whole sequence (Springer et al. 2024, "Repetition improves LM
              embeddings").  ~2x tokens.
  * "bidir":  two passes, on seq and on reversed seq; states concatenated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

AMINO = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj"]
PREFIX = "Protein amino-acid sequence:"
SEPARATOR = "\nThe same sequence again:"


def load_text_backbone(path: str, dtype: torch.dtype, random_init_seed: int | None = None) -> nn.Module:
    """Return the text decoder (no LM head, no vision tower) of a Qwen checkpoint.

    random_init_seed: build the same architecture with freshly initialised weights (control for
    "does language pre-training help?").  The frozen random weights are not saved, so the seed is
    stored in the checkpoint and the same weights are rebuilt on load (same torch version, CPU init)."""
    cfg = AutoConfig.from_pretrained(path)
    multimodal = hasattr(cfg, "text_config")  # Qwen3.5 checkpoints
    if random_init_seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_init_seed)
            if multimodal:
                from transformers import AutoModelForImageTextToText
                full = AutoModelForImageTextToText.from_config(cfg)
                backbone = full.model.language_model
                del full
            else:
                from transformers import AutoModel
                backbone = AutoModel.from_config(cfg)
        return backbone.to(dtype)
    if multimodal:
        from transformers import AutoModelForImageTextToText
        full = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
        backbone = full.model.language_model
        del full
    else:
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(path, dtype=dtype)
    return backbone


class QwenProteinEncoder(nn.Module):
    def __init__(self, qwen_path: str, out_dim: int = 1024, mode: str = "echo",
                 lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                 lora_targets: list[str] | None = None, dtype: torch.dtype = torch.bfloat16,
                 adapter_hidden: int = 2048, gradient_checkpointing: bool = True,
                 lora_compute: str = "fp32", esm_inject: str | None = None, esm_inject_hidden: int = 2048,
                 random_init_seed: int | None = None):
        """esm_inject: path of an ESM2 checkpoint whose (frozen) residue features are added, through a trained
        zero-initialised projector (esm_encoder.ESMFeaturizer), to the Qwen input embeddings at every residue
        token (Calpha-style).  random_init_seed: use a randomly initialised Qwen (see load_text_backbone).
        lora_compute: how LoRA adapters run inside the bf16 backbone.
          "fp32"     PEFT default: adapter weights fp32; every LoRA call casts its input to fp32 (slow).
          "bf16"     adapter weights stored in bf16 (fastest; optimizer updates in bf16 precision).
          "autocast" adapter weights kept fp32 (master copy), but the backbone runs under bf16
                     autocast and PEFT's input casting is disabled, so LoRA matmuls run in bf16.
        """
        super().__init__()
        assert mode in ("causal", "echo", "bidir")
        assert lora_compute in ("fp32", "bf16", "autocast")
        self.lora_compute = lora_compute
        self.mode = mode
        self.qwen_path = qwen_path
        self.tok = AutoTokenizer.from_pretrained(qwen_path)
        self.residue_ids = self._residue_token_ids()
        self.prefix_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.sep_ids = self.tok.encode(SEPARATOR, add_special_tokens=False)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

        backbone = load_text_backbone(qwen_path, dtype, random_init_seed)
        backbone.requires_grad_(False)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.config.use_cache = False
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            backbone = get_peft_model(backbone, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=lora_targets or DEFAULT_LORA_TARGETS, bias="none"),
                autocast_adapter_dtype=(lora_compute != "bf16"))
            if lora_compute == "autocast":
                from peft.tuners.tuners_utils import BaseTunerLayer
                for m in backbone.modules():
                    if isinstance(m, BaseTunerLayer):
                        m.cast_input_dtype_enabled = False
        self.backbone = backbone
        n_layers = self._base().config.num_hidden_layers
        hidden = self._base().config.hidden_size
        self.esm_inject = None
        if esm_inject:
            from .esm_encoder import ESMFeaturizer
            self.esm_inject = ESMFeaturizer(esm_inject, out_dim=hidden, hidden=esm_inject_hidden, dtype=dtype)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers + 1))
        in_dim = 2 * hidden if mode == "bidir" else hidden
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, adapter_hidden), nn.GELU(),
            nn.Linear(adapter_hidden, out_dim))
        self.hparams = dict(out_dim=out_dim, mode=mode, lora_r=lora_r, lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout, lora_targets=lora_targets, adapter_hidden=adapter_hidden,
                            lora_compute=lora_compute, esm_inject=esm_inject, esm_inject_hidden=esm_inject_hidden,
                            random_init_seed=random_init_seed)

    # ------------------------------------------------------------------
    def _base(self):
        b = self.backbone
        return b.get_base_model() if hasattr(b, "get_base_model") else b

    def _residue_token_ids(self) -> dict[str, int]:
        ids = {}
        for a in AMINO:
            t = self.tok.encode(" " + a, add_special_tokens=False)
            if len(t) != 1:
                t = self.tok.encode(a, add_special_tokens=False)
            assert len(t) == 1, f"residue {a} is not a single token"
            ids[a] = t[0]
        assert len(set(ids.values())) == len(ids)
        return ids

    def _encode_residues(self, seq: str) -> list[int]:
        return [self.residue_ids.get(c, self.residue_ids["X"]) for c in seq]

    def _build(self, seqs: list[str], reverse: bool = False):
        """input_ids/attention_mask (right padded) and index of residue states."""
        rows, gather = [], []
        for s in seqs:
            s = s[::-1] if reverse else s
            res = self._encode_residues(s)
            if self.mode == "echo":
                ids = self.prefix_ids + res + self.sep_ids + res
            else:
                ids = self.prefix_ids + res
            start = len(ids) - len(res)
            rows.append(ids)
            gather.append(list(range(start, start + len(res))))
        T = max(len(r) for r in rows)
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(rows), T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        idx = torch.zeros((len(rows), L), dtype=torch.long)
        for i, (r, g) in enumerate(zip(rows, gather)):
            input_ids[i, : len(r)] = torch.tensor(r)
            attn[i, : len(r)] = 1
            idx[i, : len(g)] = torch.tensor(g)
        return input_ids, attn, idx

    def _inject_positions(self, seqs: list[str], idx: torch.Tensor) -> list[torch.Tensor]:
        """Token positions of every residue copy (echo has two copies; the 2nd is idx)."""
        if self.mode != "echo":
            return [idx]
        first = torch.zeros_like(idx)
        for i, s in enumerate(seqs):
            first[i, : len(s)] = torch.arange(len(self.prefix_ids), len(self.prefix_ids) + len(s))
        return [first, idx]

    def _run(self, input_ids, attn, inj):
        if inj is None:
            return self.backbone(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        feats, positions = inj  # feats: (B, L, H), zero at padding (so padded idx 0 adds nothing)
        emb = self._base().get_input_embeddings()(input_ids)
        for pos in positions:
            emb = emb.scatter_add(1, pos.unsqueeze(-1).expand(-1, -1, emb.shape[-1]), feats.to(emb.dtype))
        return self.backbone(inputs_embeds=emb, attention_mask=attn, output_hidden_states=True)

    def _states(self, input_ids, attn, idx, inj=None):
        if self.lora_compute == "autocast":
            with torch.autocast(input_ids.device.type, dtype=torch.bfloat16):
                out = self._run(input_ids, attn, inj)
        else:
            out = self._run(input_ids, attn, inj)
        w = self.layer_logits.softmax(0)
        mixed = None
        for i, h in enumerate(out.hidden_states):
            g = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])).float()
            mixed = w[i] * g if mixed is None else mixed + w[i] * g
        return mixed  # (B, L, H)

    def forward(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.layer_logits.device
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        L = int(lens.max())
        mask = (torch.arange(L, device=dev)[None] < lens[:, None]).long()
        ids, attn, idx = (t.to(dev) for t in self._build(seqs))
        feats = self.esm_inject(seqs) if self.esm_inject is not None else None
        inj = None if feats is None else (feats, self._inject_positions(seqs, idx))
        h = self._states(ids, attn, idx, inj)
        if self.mode == "bidir":
            ids_r, attn_r, idx_r = (t.to(dev) for t in self._build(seqs, reverse=True))
            inj_r = None
            if feats is not None:  # features of the reversed sequence = per-sequence reversal
                j = torch.arange(L, device=dev)[None].expand(len(seqs), -1)
                src = (lens[:, None] - 1 - j).clamp(min=0)
                feats_r = torch.gather(feats, 1, src.unsqueeze(-1).expand(-1, -1, feats.shape[-1])) * mask.unsqueeze(-1)
                inj_r = (feats_r, self._inject_positions(seqs, idx_r))
            h_r = self._states(ids_r, attn_r, idx_r, inj_r)
            # un-reverse each sequence: position j of the original = position len-1-j of the reversed run
            j = torch.arange(L, device=dev)[None].expand(len(seqs), -1)
            src = (lens[:, None] - 1 - j).clamp(min=0)
            h_r = torch.gather(h_r, 1, src.unsqueeze(-1).expand(-1, -1, h_r.shape[-1]))
            h = torch.cat([h, h_r], dim=-1)
        s_lm = self.adapter(h) * mask.unsqueeze(-1)
        return s_lm, mask

    # ------------------------------------------------------------------
    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def save(self, path: str, extra: dict | None = None):
        torch.save({"qwen_path": self.qwen_path, "hparams": self.hparams,
                    "state_dict": self.trainable_state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, dtype: torch.dtype = torch.bfloat16, map_location="cpu") -> "QwenProteinEncoder":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        hp = dict(ck["hparams"])
        enc = cls(ck["qwen_path"], dtype=dtype, gradient_checkpointing=False, **hp)
        missing = enc.load_state_dict(ck["state_dict"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        return enc


def distill_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 target_std: torch.Tensor | None = None, cos_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    """MSE in per-dimension standardised units + cosine term, over valid residues."""
    m = mask.bool()
    p, t = pred[m], target[m].float()
    if target_std is not None:
        p, t = p / target_std, t / target_std
    mse = F.mse_loss(p, t)
    cos = 1 - F.cosine_similarity(p, t, dim=-1).mean()
    return mse + cos_weight * cos, {"mse": mse.item(), "cos": 1 - cos.item()}
