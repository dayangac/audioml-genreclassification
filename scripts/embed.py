"""Embed each clip with AST as five 10 s windows; writes data/gtzan/embeddings/ast_{split}.npz."""

from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gtzan"

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
CLIP_S, WIN_S, HOP_S = 30, 10, 5


def windows(y, sr):
    """Cut a clip (padded/cut to CLIP_S) into WIN_S windows every HOP_S seconds."""
    y = librosa.util.fix_length(y, size=CLIP_S * sr)
    win, hop = WIN_S * sr, HOP_S * sr
    return [y[s:s + win] for s in range(0, len(y) - win + 1, hop)]


def load_model(device):
    """Return an embed(list of windows) -> (n_windows, 768) tensor function."""
    from transformers import ASTFeatureExtractor, ASTModel
    model = ASTModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ASTFeatureExtractor.from_pretrained(MODEL_ID)

    def embed(batch):
        inp = proc(batch, sampling_rate=SR, return_tensors="pt").to(device)
        return model(**inp).pooler_output
    return embed


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embed = load_model(device)
    sr = SR
    out = DATA / "embeddings"
    out.mkdir(exist_ok=True)
    for split in ("train", "valid", "test"):
        ref = np.load(DATA / "features" / f"{split}.npz")
        embs = []
        with torch.no_grad():
            for i, rel in enumerate(ref["paths"]):
                y, _ = librosa.load(DATA / "genres" / str(rel), sr=sr, mono=True)
                embs.append(embed(windows(y, sr)).float().cpu().numpy())
                if i % 100 == 0:
                    print(f"{split} {i}/{len(ref['paths'])}", flush=True)
        np.savez_compressed(out / f"ast_{split}.npz", emb=np.stack(embs),
                            labels=ref["labels"], paths=ref["paths"])
        print(f"{split}: {np.stack(embs).shape}", flush=True)


if __name__ == "__main__":
    main()
