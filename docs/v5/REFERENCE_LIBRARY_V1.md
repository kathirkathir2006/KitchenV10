# Reference library v1

- **Source:** `kiq_v5_real_photo_v1`, train split only (791 images). The builder never opens val or
  test images, and the freeze script checks every reference against the photo-set manifest.
- **Kaggle run:** `kathiresannatarajan/kiq-v5-reflib-v1`, version 1.
- **Result:** 524 references across 27 labels (26 foods plus non_food).
- **Manifest sha:** `7e95ee5636af0a14b2839b1a242699bc02635b5d0956c6d6e89a5b9019fc0a02`
  (frozen in `reference_library_v1/`).

## Selection criteria

All steps use frozen DINOv2 ViT-B/14 CLS embeddings and train images only.

1. **Label consistency.** A train image is eligible if either of these holds:
   - its own label's centroid, computed without that image, is the nearest centroid; or
   - the majority of its 5 nearest train neighbours share its label.

   Images that fail both are treated as likely mislabelled and excluded. If fewer than 5 images
   in a label pass, the label falls back to its 5 images closest to the centroid. No label needed
   this fallback.
2. **Deduplication.** Within a label, images with cosine ≥ 0.90 to an already kept image are
   dropped. Candidates are taken in order of closeness to the centroid.
3. **Representative cap.** Each label keeps at most 20 references. The medoid is chosen first,
   then farthest-point sampling over the eligible pool, so references cover the label's spread
   rather than piling up near the centre.

Every reference records:

- its stable id `rpv1:<openverse_id>` and sha256;
- label, family and state;
- photographer group, provider, landing URL, licence and licence URL;
- the selection evidence: own-centroid cosine, kNN majority, nearest centroid, and whether it
  passed the consistency check.

Labels below the 20 cap: salad 15 (only 16 of 30 passed consistency), packaged_cheese 13,
tomato_paste 17, rice 19.

Embeddings are tied to one model version. Adapter C re-embeds the same reference images in its
own space; the reference set itself does not change.
