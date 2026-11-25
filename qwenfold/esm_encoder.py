"""Small ESM2 (35M / 150M / 650M) as a drop-in replacement for ESMFold's ESM2-3B.

    sequence -> ESM2-small (frozen, optional LoRA) -> learned layer mix -> adapter MLP -> s_lm (L x 1024)

Same interface as QwenProteinEncoder (forward(seqs) -> (s_lm, mask), save / load /
trainable_state_dict), so train_distill.py / predict.py can use either.  ESM2 is
bidirectional, so there is no context "mode".  This is the protein-specific
baseline of the same size class as the general-purpose LLM (README question 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

ESM_LORA_TARGETS = ["query", "key", "value", "dense"]


class ESMProteinEncoder(nn.Module):
    encoder_type = "esm2"

    def __init__(self, esm_path: str, out_dim: int = 1024, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, lora_targets: list[str] | None = None,
                 dtype: torch.dtype = torch.bfloat16, adapter_hidden: int = 2048,
                 gradient_checkpointing: bool = False, lora_compute: str = "autocast", mode: str = "bidir"):
        super().__init__()
        assert lora_compute in ("fp32", "bf16", "autocast")
        self.lora_compute = lora_compute
        self.mode = "esm2"
        self.esm_path = esm_path
        self.tok = AutoTokenizer.from_pretrained(esm_path)
        backbone = EsmModel.from_pretrained(esm_path, dtype=dtype, add_pooling_layer=False)
        backbone.requires_grad_(False)
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            backbone = get_peft_model(backbone, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=lora_targets or ESM_LORA_TARGETS, bias="none"),
                autocast_adapter_dtype=(lora_compute != "bf16"))
            if lora_compute == "autocast":
                from peft.tuners.tuners_utils import BaseTunerLayer
                for m in backbone.modules():
                    if isinstance(m, BaseTunerLayer):
                        m.cast_input_dtype_enabled = False
        self.backbone = backbone
        cfg = self._base().config
        self.layer_logits = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.adapter = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, adapter_hidden), nn.GELU(),
            nn.Linear(adapter_hidden, out_dim))
        self.hparams = dict(out_dim=out_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                            lora_targets=lora_targets, adapter_hidden=adapter_hidden, lora_compute=lora_compute)

    def _base(self):
        b = self.backbone
        return b.get_base_model() if hasattr(b, "get_base_model") else b

    def _build(self, seqs: list[str]):
        """<cls> residues <eos>, right padded; residue i sits at token i+1."""
        T = max(len(s) for s in seqs) + 2
        ids = torch.full((len(seqs), T), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        unk = self.tok.unk_token_id
        for i, s in enumerate(seqs):
            res = [self.tok.convert_tokens_to_ids(c) for c in s]
            res = [unk if r is None else r for r in res]
            row = [self.tok.cls_token_id] + res + [self.tok.eos_token_id]
            ids[i, : len(row)] = torch.tensor(row)
            attn[i, : len(row)] = 1
        return ids, attn

    def forward(self, seqs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.layer_logits.device
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        L = int(lens.max())
        mask = (torch.arange(L, device=dev)[None] < lens[:, None]).long()
        ids, attn = (t.to(dev) for t in self._build(seqs))
        if self.lora_compute == "autocast" and dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.backbone(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        else:
            out = self.backbone(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        w = self.layer_logits.softmax(0)
        h = None
        for i, hs in enumerate(out.hidden_states):
            g = hs[:, 1: L + 1].float()
            h = w[i] * g if h is None else h + w[i] * g
        s_lm = self.adapter(h) * mask.unsqueeze(-1)
        return s_lm, mask

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def save(self, path: str, extra: dict | None = None):
        torch.save({"encoder": self.encoder_type, "esm_path": self.esm_path, "hparams": self.hparams,
                    "state_dict": self.trainable_state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, dtype: torch.dtype = torch.bfloat16, map_location="cpu") -> "ESMProteinEncoder":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        enc = cls(ck["esm_path"], dtype=dtype, gradient_checkpointing=False, **ck["hparams"])
        missing = enc.load_state_dict(ck["state_dict"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        return enc


class ESMFeaturizer(nn.Module):
    """Frozen ESM2 -> learned layer mix -> MLP (last layer zero-initialised) -> per-residue vectors.

    Used to inject protein-LM knowledge into a general LLM at the amino-acid token positions
    (Calpha-style, arXiv:2611.20986: "each residue = its letter plus what ESM2 knows about it").
    Only layer_logits and proj are trained; the ESM2 weights stay frozen (and are not saved).
    Zero initialisation means training starts from the plain LLM.
    """

    def __init__(self, esm_path: str, out_dim: int, hidden: int = 2048, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.esm_path = esm_path
        self.tok = AutoTokenizer.from_pretrained(esm_path)
        self.esm = EsmModel.from_pretrained(esm_path, dtype=dtype, add_pooling_layer=False)
        self.esm.requires_grad_(False)
        self.esm.eval()
        cfg = self.esm.config
        self.layer_logits = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.proj = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, hidden), nn.GELU(),
                                  nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.esm.eval()  # frozen: never use dropout
        return self

    def forward(self, seqs: list[str]) -> torch.Tensor:
        """(B, L, out_dim) for residues 0..len-1, zeros at padding."""
        dev = self.layer_logits.device
        lens = torch.tensor([len(s) for s in seqs], device=dev)
        L = int(lens.max())
        ids, attn = (t.to(dev) for t in build_esm_batch(self.tok, seqs))
        with torch.no_grad():
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hs = self.esm(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
            else:
                hs = self.esm(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        w = self.layer_logits.softmax(0)
        h = None
        for i, x in enumerate(hs):
            g = x[:, 1: L + 1].float()
            h = w[i] * g if h is None else h + w[i] * g
        mask = (torch.arange(L, device=dev)[None] < lens[:, None]).unsqueeze(-1)
        return self.proj(h) * mask


def build_esm_batch(tok, seqs: list[str]):
    """<cls> residues <eos>, right padded; residue i sits at token i+1."""
    T = max(len(s) for s in seqs) + 2
    ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        res = [tok.convert_tokens_to_ids(c) for c in s]
        res = [tok.unk_token_id if r is None else r for r in res]
        row = [tok.cls_token_id] + res + [tok.eos_token_id]
        ids[i, : len(row)] = torch.tensor(row)
        attn[i, : len(row)] = 1
    return ids, attn


def load_encoder(path: str, dtype: torch.dtype = torch.bfloat16):
    """Load any encoder checkpoint (Qwen checkpoints have no 'encoder' key)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("encoder") == "esm2":
        return ESMProteinEncoder.load(path, dtype=dtype)
    from .qwen_encoder import QwenProteinEncoder
    return QwenProteinEncoder.load(path, dtype=dtype)
