# Real-photo set v1: specification

Builder: `kaggle/kernels/kiq-v5-real-photo-v1/kiq_v5_real_photo_v1.py` (Kaggle, run once).
Images stay on Kaggle. The repo holds the manifest, build report and attribution.

| Requirement | How it is enforced |
|---|---|
| Provenance and licence | Every item stores its Openverse id, provider, landing URL, file URL, creator, licence, licence version, licence URL and attribution. Only `cc0` and `by` are accepted. Items with no licence URL or landing page are rejected. |
| No overlap with training/reference images | Candidates are rejected on any of: an id or URL that appears in a mounted training/reference manifest; a sha256 or dHash (Hamming ≤ 8) match to any mounted training/reference image; DINOv2 cosine ≥ 0.92 against those images and the 108 library embeddings. |
| No benchmark leakage | Candidates are rejected on a sha256 match, dHash ≤ 8, or DINOv2 cosine ≥ 0.92 against fingerprints of all 41 scenes and 47 ground-truth crops. The fingerprints (`kiq-v5-benchmark-fingerprints`) contain no images. Each rejection is logged with the nearest scene. |
| No duplicates inside the set | Within the set, DINOv2 cosine ≥ 0.95 or dHash ≤ 6 counts as a duplicate. The first item in deterministic order is kept. |
| Fixed once created | `manifest.json` records a sha256 per image. The manifest's own sha is committed in `docs/v5/`. The builder refuses to overwrite. Every consumer must re-hash every image and abort on any mismatch. Any change becomes v2, never an edit of v1. |
| Separate roles | Split is decided by `sha256(set_id, creator group)`: 60% train, 20% val, 20% test. A photographer's images never span splits (asserted). Quotas per known identity are 30 train / 10 val / 10 test. |
| Open-set | red_bell_pepper, flatbread, beef_stew, dumplings and pancakes appear only in val/test, as `open_set_unknown`. Training of these is asserted to be absent. |

How each role may be used:

- **train:** adapter fitting and reference library construction.
- **val:** thresholds, loss weights and model selection.
- **test:** reported once per frozen candidate, never used for selection.
- **The 41 scenes:** reported, with the photo subset gated separately. Never used for fitting or
  selection.

## Known limitation: label verification

Labels are verified by a title/tag keyword match plus open-weights CLIP agreement. CLIP must rank
the label in its top 3 for train and first for val/test. CLIP is used only for curation, never as
the scanner.

This makes val/test labels machine-verified rather than human-verified. It also biases val/test
toward images CLIP finds unambiguous, so test scores may be optimistic on hard cases. These
images are web photos, not kitchen-camera captures.
