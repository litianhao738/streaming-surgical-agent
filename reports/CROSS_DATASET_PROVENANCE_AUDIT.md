l# CholecTrack20 Cross-Dataset Provenance Audit

Access date: `2026-08-21`. Audit mode: read-only.

## Decision

| Item | Status |
| --- | --- |
| Video provenance | **PASS_OFFICIAL_ID_RELATION** |
| Ontology | **PARTIAL_RECOVERED_WITH_EXCEPTIONS** |
| Frame index rule | **UNRESOLVED_FORMAL_RULE_STRONGLY_SUPPORTED** |
| VID30 | **BLOCKED_VID17_CORE_ANNOTATION_DUPLICATE_CONFIRMED** |
| VID31 | **BLOCKED_ANNOTATION_MEDIA_PAIRING** |
| Negative labels | **PARTIAL** |
| IVT conflicts | **PARTIAL_32_EXPLAINED_2_UNRESOLVED** |
| Formal experiments | **BLOCKED** |

Formal experiments remain blocked. The audit recovers useful provenance and
ontology evidence but does not authorize data repair or label rewriting.
+
## Video Provenance

All 20 Track20 IDs map to an upstream collection: 15 map to Cholec80, 14 map to CholecT50, and 9 occur in both. This is based on the official preserved-ID convention, not on pixel hashes.

| Track20 | Split | Cholec80 | CholecT50 |
| --- | --- | --- | --- |
| VID01 | testing | video01 | VID01 (train) |
| VID02 | training | video02 | VID02 (train) |
| VID04 | training | video04 | VID04 (train) |
| VID06 | testing | video06 | VID06 (test) |
| VID07 | testing | video07 | - |
| VID11 | training | video11 | - |
| VID12 | testing | video12 | VID12 (val) |
| VID13 | training | video13 | VID13 (train) |
| VID17 | training | video17 | - |
| VID23 | training | video23 | VID23 (train) |
| VID25 | testing | video25 | VID25 (train) |
| VID30 | validation | video30 | - |
| VID31 | training | video31 | VID31 (train) |
| VID37 | training | video37 | - |
| VID39 | testing | video39 | - |
| VID92 | testing | - | VID92 (train) |
| VID96 | training | - | VID96 (train) |
| VID103 | training | - | VID103 (train) |
| VID110 | validation | - | VID110 (train) |
| VID111 | testing | - | VID111 (test) |

## Ontology

- Instrument IDs are verified from Track20 JSON and official Cholec80/CholecT50 metadata. Track20 adds ID 6 `specimen-bag`; CholecT50 IVT has only instruments 0-5.
- Verb, target, and 100 triplet names were recovered from the official CholecT50 challenge validation JSON and its `label_mapping.txt`.
- Instrument: 0=grasper, 1=bipolar, 2=hook, 3=scissors, 4=clipper, 5=irrigator, 6=specimen-bag.
- Verb: 0=grasp, 1=retract, 2=dissect, 3=coagulate, 4=clip, 5=cut, 6=aspirate, 7=irrigate, 8=pack, 9=null_verb.
- Target: 0=gallbladder, 1=cystic_plate, 2=cystic_duct, 3=cystic_artery, 4=cystic_pedicle, 5=blood_vessel, 6=fluid, 7=abdominal_wall_cavity, 8=liver, 9=adhesion, 10=omentum, 11=peritoneum, 12=gut, 13=specimen_bag, 14=null_target.
- The complete 100-class Triplet map is retained in the machine-readable audit.
- Of 22,994 comparable Track20 instances, 22,960 match the upstream IVT relation and 34 do not.
- Track20 phase names are documented, but the numeric mapping remains `UNRESOLVED`: the paper does not publish an explicit ID table and CholecT50 orders IDs 2/3 differently from the Track20 paper's phase list.

## The 34 IVT Conflicts

- 32 rows are explained by semantic scope: Track20 boxes the specimen bag (instrument 6), while triplet 12 means `grasper, grasp, specimen_bag`.
- One VID25 row has triplet 43 (`bipolar, retract, gallbladder`) but stores instrument 0 (`grasper`): `UNRESOLVED`.
- One VID23 row keeps triplet 94 but stores verb/target as `-1/-1` instead of upstream null IDs `9/14`: `UNRESOLVED_NULL_ENCODING`.

## Frame Index Rule

All Track20 annotation IDs are `1 mod 25`; the Cholec80 official extractor is zero-based and its 1 FPS `Original_frame_id` values are `0 mod 25`. Therefore `decoder_frame_index = annotation_frame_id - 1` is strongly supported. It is still marked `UNRESOLVED` for formal evaluation because same-VID upstream pixels were not available for hash comparison and release-specific phase labels are not identical.

Six Cholec80-derived test videos support the provenance chain; VID92 and VID111 lack an accessible same-VID upstream frame reference. The candidate index is valid inside all eight local MP4 containers, but no automatic offset was written to config.

## VID30 / VID31

- VID30 JSON is not merely misaligned: its 1,863 instances have the same frame IDs and identical geometry, instrument, operator, visibility and three track-ID fields as VID17. Only phase and several scene-condition fields differ. It is therefore confirmed as a VID17 core-annotation duplicate; the audit does not infer how that duplication occurred.
- VID30 PNG IDs exactly equal VID31 annotation IDs. The public issue reports that VID31 annotation appears to describe VID30, but there is no maintainer confirmation or upstream pixel reference; that pairing remains blocked rather than auto-repaired.
- VID31 PNG identity and the missing correct VID31 annotation remain unresolved.

## Negative Labels

Triplet `-2` occurs 1,129 times; 1,116 are specimen-bag instances, which have no CholecT50 instrument/triplet class. The remaining 13 cases remain unresolved. Categorical `-1` is not equivalent to CholecT50's explicit null classes and remains an invalid/missing sentinel only.

## Remaining Blockers

- Maintainer-provided corrected VID30/VID31 annotations and media identities.
- Explicit Track20 phase ID table and negative-sentinel specification.
- Official Track20 annotation-key to decoder-index rule, or same-VID upstream frames for pixel-level verification.
- Source CholecT50 annotations for overlapping Track20 VIDs; the public validation package contains only VID68/70/73/74/75, none of which is in Track20.

## Provenance

- [CAMMA dataset-overlap CholecT50 split](https://raw.githubusercontent.com/CAMMA-public/camma_dataset_overlaps/8347b9f4cb02ebe739747903e6eada272ee9d25e/resources/CholecT50_splits.json), version `8347b9f4cb02ebe739747903e6eada272ee9d25e`: CholecT50 membership and official split
- [CAMMA dataset-overlap Cholec80 split](https://raw.githubusercontent.com/CAMMA-public/camma_dataset_overlaps/8347b9f4cb02ebe739747903e6eada272ee9d25e/resources/Cholec80_splits.json), version `8347b9f4cb02ebe739747903e6eada272ee9d25e`: Cholec80 video membership and split
- [CholecT50 official format](https://github.com/CAMMA-public/cholect50/blob/354020eb217086a07be4b225d803c7fea7760b7a/docs/README-Format.md), version `354020eb217086a07be4b225d803c7fea7760b7a`: 1 FPS sequential image IDs, ontology and mapping format
- [CholecT50 official preserved-ID statement](https://github.com/CAMMA-public/cholect50/blob/354020eb217086a07be4b225d803c7fea7760b7a/docs/README-Splits.md), version `354020eb217086a07be4b225d803c7fea7760b7a`: Numeric video IDs are consistent across CAMMA datasets
- [CholecT50 challenge mapping](https://s3.unistra.fr/camma_public/datasets/cholect50/CholecT50-Challenge-Validation.zip#cholect50-challenge-val/label_mapping.txt), version `Mon, 20 Feb 2023 13:09:26 GMT`: 100 IVT-to-I/V/T relations
- [CholecT50 challenge ontology](https://s3.unistra.fr/camma_public/datasets/cholect50/CholecT50-Challenge-Validation.zip#cholect50-challenge-val/labels/VID68.json), version `Mon, 20 Feb 2023 13:09:26 GMT`: Instrument, verb, target, triplet and phase ID names
- [CholecT50 challenge README](https://s3.unistra.fr/camma_public/datasets/cholect50/CholecT50-Challenge-Validation.zip#cholect50-challenge-val/README.md), version `Mon, 20 Feb 2023 13:09:26 GMT`: Release identity, source institution and preserved video IDs
- [Cholec80 official frame extractor](https://github.com/CAMMA-public/SelfSupSurg/blob/8a03d28948e59471cc8ea865d56632eda048079c/utils/extract_frames_ch80.py), version `8a03d28948e59471cc8ea865d56632eda048079c`: 25 FPS check and zero-based decoder-frame filenames
- [Cholec80 official 1 FPS labels](https://s3.unistra.fr/camma_public/github/selfsupsurg/ch80_labels.zip), version `Mon, 06 Feb 2023 12:27:36 GMT`: Original_frame_id values and phase labels for 80 videos
- [CholecTrack20 official README](https://github.com/CAMMA-public/cholectrack20/blob/6406b52806b6d881ef5ee9bf75255b3d0af665f9/README.md), version `6406b52806b6d881ef5ee9bf75255b3d0af665f9`: Dataset overlap, preserved identities, 1 FPS labels and 25 FPS videos
- [CholecTrack20 CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Nwoye_CholecTrack20_A_Multi-Perspective_Tracking_Dataset_for_Surgical_Tools_CVPR_2025_paper.pdf), version `CVPR 2025`: Track20 phase names and dataset design
- [CholecTrack20 public issue 10](https://github.com/CAMMA-public/cholectrack20/issues/10), version `opened 2026-04-22; no maintainer resolution at access date`: Independent report that VID31 annotation appears to describe VID30

No training/inference code or source dataset file was modified.
