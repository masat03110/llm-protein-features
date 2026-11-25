"""Minimal, dependency-free PDB reader used by the scorer.

Residues are keyed by (residue_number, insertion_code) so that a model and a
native structure that share CASP numbering can be matched residue-by-residue
(sequence-dependent evaluation, as in CASP).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K", "UNK": "X",
}


@dataclass
class Residue:
    resname: str
    atoms: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Structure:
    residues: dict[tuple[int, str], Residue]  # ordered (insertion order)

    @property
    def keys(self) -> list[tuple[int, str]]:
        return list(self.residues.keys())

    def sequence(self) -> str:
        return "".join(THREE_TO_ONE.get(r.resname, "X") for r in self.residues.values())

    def ca_coords(self, keys=None) -> np.ndarray:
        keys = self.keys if keys is None else keys
        return np.stack([self.residues[k].atoms["CA"] for k in keys])


def read_pdb(path: str, chain: str | None = None) -> Structure:
    """Read the first model of a PDB file.

    * Hydrogens are dropped; for alternate locations the first one is kept.
    * Residues without a CA atom are dropped (they cannot be superposed).
    * If ``chain`` is None, the first chain encountered is used.
    """
    residues: dict[tuple[int, str], Residue] = {}
    chosen_chain = chain
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec.startswith("ENDMDL"):
                break
            if rec not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            if rec == "HETATM" and resname != "MSE":
                continue
            ch = line[21]
            if chosen_chain is None:
                chosen_chain = ch
            if ch != chosen_chain:
                continue
            altloc = line[16]
            if altloc not in (" ", "A", "1"):
                continue
            atom = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if element == "H" or (not element and atom.startswith("H")):
                continue
            if resname == "MSE" and atom == "SE":
                atom = "SD"
                resname = "MET"
            key = (int(line[22:26]), line[26].strip())
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            res = residues.setdefault(key, Residue(resname=resname))
            res.atoms.setdefault(atom, xyz)
    residues = {k: r for k, r in residues.items() if "CA" in r.atoms}
    return Structure(residues=residues)
