# Notes: ideas to make this novel

## Latent space functional mapping
- Treat each model's latent space as a space and learn a **functional map** between them, as in shape correspondence from geometry processing: a small matrix that maps one space's basis functions onto the other's.
- Build a kNN graph per model on its embeddings of the same clips. Compute Laplacian eigenvectors on each graph and fit a functional map C between the two eigenbases.
- What C shows:
  - near diagonal means the two models organise the clips the same way;
  - off-diagonal blocks show where they disagree (e.g. rock, disco).
- Compare against CKA as the standard similarity measure; the functional map gives structure, CKA only a single number.
- Controls: maps between seeds of the same model first, then CNN against GNN.

## GDL (geometric deep learning)
- **Tonnetz GNN** on chroma: 12 pitch-class nodes, edges for fifths and major/minor thirds. Run it as a hybrid with timbre input, since chroma alone is weak (feature analysis).
- **Transposition test:** shifting by k semitones is an exact roll for chroma; use `librosa.effects.pitch_shift` for mel. Measure how each latent space moves.
- With functional maps: transposition acting on the latent space should be a structured map for the Tonnetz model and not for the CNN.

## Pitch
"Geometry-processing tools (functional maps) for comparing what music models learn, tested on a model with music-theory geometry built in."

## Remember
- Check course rules on publishing / reuse before sharing anything.
- Use the fault-filtered split throughout; seeds ≥ 3; test once.
- Current numbers (test): AST + LDA + kNN 0.855, AST + kNN 0.803; handcrafted: SVM 0.624, LogReg 0.610, LDA + kNN 0.586, kNN 0.510.
- AST was trained on AudioSet, whose labels include genres and whose YouTube clips may contain GTZAN songs: say so in the write-up.
- CLAP (`laion/larger_clap_music`) gives collapsed embeddings under transformers 5.22 (zero-shot 15%); not used.
- AST latent space is a ready third "model" for the functional-map comparison.
