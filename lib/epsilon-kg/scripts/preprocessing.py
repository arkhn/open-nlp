import argparse
from quickumls import QuickUMLS
import json
import math
import os
from pathlib import Path
from typing import List, Tuple
from autodp.mechanism_zoo import GaussianMechanism
from autodp.composition import Composition

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
import torch

# ------------------------- Phase 0 -------------------------


def load_data(parquet_path: Path, text_col: str = "text") -> pd.Series:
    """Read Parquet and return the text column."""
    return pd.read_parquet(parquet_path, columns=[text_col])[text_col]


# ------------------------- Phase 1 -------------------------


def extract_keyphrases(docs: pd.Series) -> List[List[str]]:
    """Extract keyphrases with QuickUMLS (path from $QUICKUMLS_DIR)."""
    qdir = os.getenv("QUICKUMLS_DIR")
    matcher = QuickUMLS(qdir)  # default threshold
    kp_docs: List[List[str]] = []
    for doc in docs:
        matches = matcher.match(doc, best_match=True, ignore_syntax=False)
        phrases = {m["ngram"] for m in matches}
        kp_docs.append(sorted(phrases))
    return kp_docs


# ------------------------- Phase 2 -------------------------


def embed_keyphrases(
    kp_lists: List[List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    vocab = sorted({p for lst in kp_lists for p in lst})
    emb = model.encode(
        vocab, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(emb), vocab


# ------------------------- Phase 3 -------------------------


def _sigma_from_eps(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """σ for Gaussian mech. from (ε,δ) with L2 sensitivity 1 (Dwork 2014)."""
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / eps


def build_dp_kde(
    embeddings: np.ndarray, epsilon: float, delta: float, bandwidth: str | float = "silverman"
):
    """Return a callable dp_kde(x) with Gaussian noise calibrated to ε,δ."""
    kde = gaussian_kde(embeddings.T, bw_method=bandwidth)
    sigma = _sigma_from_eps(epsilon, delta)

    def dp_kde(vecs: np.ndarray):
        dens = kde(vecs.T) if vecs.ndim == 2 else kde(vecs)
        return dens + np.random.normal(0.0, sigma, size=dens.shape)

    return dp_kde, sigma


# ------------------------- Phase 4 -------------------------


def sample_keyphrases_dp(
    dp_kde, vocab_emb: np.ndarray, vocab: List[str], n: int = 1000
) -> List[str]:
    scores = dp_kde(vocab_emb)
    probs = np.clip(scores, 0, None)
    if probs.sum() == 0:
        raise ValueError("All DP‑KDE densities are ≤0. Increase ε or adjust bandwidth.")
    probs /= probs.sum()
    idx = np.random.choice(len(vocab), size=n, p=probs)
    return [vocab[i] for i in idx]


# ------------------------- Phase 5 -------------------------


def compute_privacy_budget(eps_vocab: float, eps_kde: float, delta: float) -> dict:
    """Advanced composition via **AutoDP** (Rényi accountant).

    Raises if AutoDP not installed.
    """
    sigma_v = _sigma_from_eps(eps_vocab, delta)
    sigma_k = _sigma_from_eps(eps_kde, delta)

    gm_v = GaussianMechanism(sigma=sigma_v, sensitivity=1.0)
    gm_k = GaussianMechanism(sigma=sigma_k, sensitivity=1.0)

    comp = Composition()
    eps_total = comp.get_eps([gm_v, gm_k], [1, 1], delta)

    return {
        "epsilon_vocab": eps_vocab,
        "epsilon_kde": eps_kde,
        "epsilon_total": eps_total,
        "delta": delta,
        "accountant": "AutoDP (Rényi)",
    }


# ------------------------- Phase 6 -------------------------


def save_samples(samples: List[str], out_path: Path):
    pd.DataFrame({"keyphrase": samples}).to_parquet(out_path, index=False)


# ------------------------- CLI & Main -------------------------


def parse_args():
    args = argparse.ArgumentParser("DP‑KPS E2E (AutoDP Rényi)")
    args.add_argument("--input", type=Path, required=True, help="Parquet with 'text' column")
    args.add_argument("--output", type=Path, default="sampled_keyphrases.parquet")
    args.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--eps_vocab", type=float, default=1.0)
    args.add_argument("--eps_kde", type=float, default=4.0)
    args.add_argument("--delta", type=float, default=1e-5)
    args.add_argument("--n_samples", type=int, default=1000)
    args.add_argument("--privacy_report", type=Path, default="privacy_account.json")
    args.add_argument("--text_col", type=str, default="text", help="Column name for text data")
    return args.parse_args()


def main():
    args = parse_args()
    docs = load_data(args.input, args.text_col)
    kp_lists = extract_keyphrases(docs)
    emb_mat, vocab = embed_keyphrases(kp_lists, args.model_name, args.batch_size)
    dp_kde, _ = build_dp_kde(emb_mat, args.eps_kde, args.delta)
    sampled = sample_keyphrases_dp(dp_kde, emb_mat, vocab, args.n_samples)

    report = compute_privacy_budget(args.eps_vocab, args.eps_kde, args.delta)
    with open(args.privacy_report, "w") as f:
        json.dump(report, f, indent=2)

    save_samples(sampled, args.output)
    print(f"✅ {len(sampled)} keyphrases saved to {args.output}")
    print(f"🔒 Privacy report → {args.privacy_report}")


if __name__ == "__main__":
    main()
