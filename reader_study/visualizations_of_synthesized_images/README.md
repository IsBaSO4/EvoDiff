# Visualizations of the Synthesized Images (Reader Study Materials)

This folder provides side-by-side visual materials of **real** and **EvoDiff-synthesized** ultrasound images, organized for expert validation of realism and pathology fidelity.

## Organization

Each subfolder corresponds to one `organ-condition` category (12 categories in total, covering 6 organs):

```
<Organ>-<Condition>/
├── real_001.png ... real_010.png            # 10 real ultrasound images
└── synthetic_001.png ... synthetic_010.png  # 10 synthesized images of the same category
```

| Organ | Conditions |
|---|---|
| Appendix | Normal, Appendicitis |
| Breast | Benign, Malignant |
| Carotid | Normal, Future Cardiovascular Diseases |
| Liver | Normal, Fatty Liver |
| Ovary | Benign, Malignant |
| Thyroid | Benign, Malignant |

## Purpose

These images accompany the reader study in which board-certified radiologists assessed (1) the realism of the synthesized images and (2) whether the synthesized images faithfully exhibit the pathological features of their assigned labels (e.g., whether a "malignant" synthetic image actually shows malignant sonographic features).

File numbering within each folder is arbitrary and carries no pairing relationship between real and synthetic images.
