# Causal Three-Frame Input Decision

Date: 2026-09-04 (Asia/Shanghai)

The current primary Pipeline uses at most three contiguous causal observations,
ordered chronologically and ending at the target frame. CholecTrack20 is
annotated at 1 FPS over 25 FPS source video, so one observation step is 25 frame
IDs. A complete window is:

```text
[t-50, t-25, t] = [t-2 seconds, t-1 second, t]
```

The target prediction stride is one annotated observation (one second); formal
evaluation does not skip otherwise eligible targets. At stream boundaries the
input contains one or two frames and the first frame is never duplicated. If
the observed frame-ID increment exceeds 25, the causal visual buffer resets:
the first post-gap target receives one image, then the window grows to two and
three images. Frames before the gap are never used to pad the window.

JointPerception and targeted Verify receive the same image tuple in every
Tracker x Gate cell. Frame selection is `fixed_all`; Tracker, Workflow and
Memory cannot select or replace images. Immediate temporal-difference features
also ignore a finalized prediction before a reset.

This supersedes the six-frame primary input contract. Existing six-frame
configs, caches and artifacts remain historical evidence and must not be mixed
with three-frame results. The prompt and JSON schema remain unchanged because
they describe the supplied causal image tuple without fixing its cardinality.
Canonical request hashing includes the selected image identities and therefore
separates three-frame calls from older six-frame cache entries.

The detector is a current-frame model and is not retrained because the VLM
window changed. The association generator is now gap-aware. Existing detector
checkpoints remain reusable; strict new Tracker artifacts should regenerate
association outputs so track identity and age do not cross a frame-clock gap.
Gate examples and fitted Gate artifacts collected with six-frame H0/Verify
requests are not reusable for the three-frame protocol.

No existing repository result establishes that six frames outperform three.
Any future frame-count comparison must use paired target identities, the same
model, prompt, schema and decoding settings, and must be completed before a
sealed Test run.
