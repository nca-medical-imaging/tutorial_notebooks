# Datasheet — 2-D microwave ablation corpus

Generated 2026-09-09T08:46:29+00:00 by
`scripts/gen_dataset.py`. **4000 instances**, 0 failures.
Splits: train 2800, val 400, test 800
(train.npz 29.2 MB, val.npz 4.2 MB, test.npz 8.6 MB).

## Motivation

To train a cellular-automaton surrogate that predicts the microwave-ablation
necrosis field fast enough to sit inside a planning loop, and to teach that
process in a 30-minute hands-on session. The ground truth is a UniPhys
Pennes-bioheat + microwave-SAR + Arrhenius cell-death simulation, which is
correct and far too slow to call inside a search.

This is the 2-D twin of ThermoNav's `cnca_indepth_v5` corpus. Channels,
normalisation, device calibration and clinical marginals are deliberately the
same so a number measured here means something next to a number measured there.

## Composition

One instance = one ablation plan on one **real patient CT slice**, extracted from ThermoNav's `cnca_indepth_v5_1mm` corpus and block-pooled to this grid.

| | |
|---|---|
| grid | 128 x 128 at 2 mm (256 mm field of view) |
| inputs | 3 channels, uint8, `[0,1]` after dequantisation |
| targets | 2 channels: Arrhenius damage fraction, and the vessel mask |
| storage | `inputs (N,3,128,128) uint8`, `targets (N,2,128,128) uint8` |
| burn time | **fixed at 300 s** — no longer an input |

Inputs, in order: `hounsfield`, `needle`, `applicator_activation`.
Targets: `cell_death`, `vessel`.

**v2 gives the model the picture, not the physics.** v1's eight inputs were the
per-voxel terms of the bioheat equation — rho*c, perfusion, k, sigma, the vessel
sink rate — which is the right input if somebody has already produced a
segmentation and a material table. v2 hands over a CT slice, a needle and a
power, and asks for the necrosis field AND the vasculature.

The vessel task is genuinely hard and the corpus is built so that it stays hard:
vessel contrast is sampled per case over 1-28 HU, which puts **22 %** of cases at
or beyond the difficulty of the real demo slice (d' = 0.17 at 2 mm, where the
best single HU threshold scores F1 0.414).

**uint8, not float32.** Both fields are in [0,1] by construction and 1/255 is far
below the disagreement between two runs of the solver at different resolutions.
It is a 4x saving on a corpus that has to reach a laptop over conference Wi-Fi.

## Ground truth

| | |
|---|---|
| solver | UniPhys 0.1.0, explicit FDM, CPU/OpenMP |
| material database | revision `4aa9e3ae7ecf99ae…` |
| geometry | 2-D anatomy **extruded** to a 32-cell slab; needles in the mid-plane; label = mid-plane of the 3-D solve |
| cooldown | protocol + 900 s, with dynamic stop |
| necrosis contour | d >= 0.99 |

**Why 3-D and sliced.** A one-cell-deep grid is a genuine 2-D heat equation and
it is the physics of an infinite slab: 90 W for 5 min gives a 40 x 48 mm lesion
where the mid-plane of the same needle in 3-D gives 40 x 42 mm. The mid-plane is
bit-identical at nz = 32, 48, 64 and 96, so the slab depth is nearly free.

**Cost of the cooldown choice.** 900 s against a fully converged
1800 s tail, over ten varied plans: 1.64x faster for a mean necrosis DSC of
0.9995 (worst 0.9978, worst disagreement 3 cells out of 782).

## Device

Medtronic Emprint HP (Thermosphere), the `mid_eff0.60` arm of ThermoNavServer's nine-arm bench fit.

| | |
|---|---|
| dial maximum | 150 W, efficiency 0.6 -> 90 W deposited |
| radiating slot | [7.25, 19.08] mm back from the tip |
| SAR penetration | 9.2506 mm (Beer-Lambert recast of the axisymmetric EM solve) |
| dielectric self-limiting | 0.85 |
| frequency / diameter | 2.45 GHz / 2.11 mm (14G) |

> **PROVISIONAL — fitted 2026-07-28 on ex vivo bovine liver bench data at 17 C, held-out RMS 9.9 %. NOT clinically validated. Every lesion in this workshop inherits that status.**

## Anatomy

**Real patient CT.** 2000 axial slices extracted from ThermoNav's
`cnca_indepth_v5` corpus at 2 mm, carrying the real Hounsfield field, the real
liver and vessel segmentations, and a real per-cell `material_id` covering bone,
kidney, muscle, skin, lung and air as well as liver and vasculature.

**The split is the corpus's PATIENT-level one**, not a cut through the slices:
train 35 patients / 1400 slices, val 5 patients / 200 slices, test 10 patients / 400 slices. Adjacent axial slices of one liver are near-duplicates, so a
random split would put a near-copy of every validation slice into training and
report a number that means nothing.

Properties are read from `material_id` through UniPhys's own table rather than
stored. They are exactly constant per id in this corpus — measured relative
spread 1e-8, which is float16 rounding — so the id is sufficient, it is smaller,
and it cannot drift from what the solver integrates.

Downsampling 1 mm to 2 mm respects what each field is: the CT is **averaged** (a
dense field), the segmentations are **max**-pooled (thin structures vanish under
averaging), and `material_id` is **mode**-pooled — averaging two material ids
gives a third material that is not present, and max-pooling gives whichever enum
ordinal happens to be largest.

*This replaced a procedural anatomy.* On synthetic slices the vessel head
collapsed onto the organ mask (precision 0.087, IoU 0.77 against the LIVER) and
no loss weighting could move it: sweeping the vessel weight over 1, 3 and 10 and
the BCE positive weight to 50 left vessel F1 pinned at 0.18 +/- 0.005 while the
necrosis DSC fell from 0.85 to 0.12. Procedural vessels ARE bright blobs, so
"bright blob" was a correct solution and there was no shape to learn.

| quantity | p5 | median | p95 | range |
|---|---|---|---|---|
| organ area (cm²) | 81.6 | 155.25 | 236.9 | 69.9–273.2 |
| vessel (% of organ) | 7.8 | 13.0 | 20.9 | 4.7–28.7 |
| vessel calibre (mm) | 4.0 | 8.0 | 28.0 | 4.0–34.2 |

The demo slice (case-1001, axial 154, de-identified segmentation) is 204 cm² and
13.9 % vessel with calibres to 26.8 mm — the procedural ranges were widened
until it sat inside them, because a demo outside its own training distribution
is a demo about extrapolation.

**Tissue properties** are UniPhys's own curated IT'IS-derived table, baked into
`ablation2d/tissue_table.json` with the database revision so the notebook needs
no compiled solver and drift is detectable:

| label | source entry | rho·c J/(mm³K) | k W/(mm K) | sigma S/mm | perfusion |
|---|---|---|---|---|---|
| background | connective tissue | 2.4353e-03 | 3.9450e-04 | 7.9196e-05 | 37.2 |
| liver | liver | 3.8190e-03 | 5.1911e-04 | 1.8504e-04 | 860.5 |
| vessel | blood | 3.7969e-03 | 5.1686e-04 | 6.6246e-04 | 1e+04 |

## Plans

| quantity | target (clinical) | achieved |
|---|---|---|
| antennas 1/2/3/4 | 55 / 25 / 13 / 7 % | 56 / 24 / 13 / 7 % |
| dial power | median 75 W, 60 % in 50-100 W, 5 % off | median 75 W, 59 %, 5.4 % off |
| duration | FIXED (not an input in v2) | 5.00 min for every needle |
| needle length | 40-200 mm | 40-200 mm, clipped to the frame |

Clinical references: KLCA 2024 practice guideline; a 2025 series of 133 lesions;
Lu et al., AJR 2002 for the 3 mm heat-sink threshold.

**Hard constraints.** Tip inside the organ and not inside a vessel lumen; the far
end of the radiating slot inside the organ; the entry point on the picture.
**Vessel crossing is deliberately permitted** — it is the heat-sink signal the
corpus exists to teach.

**The 5 % of needles at exactly 0 W are not a mistake.** They are the only
examples that teach the network that a needle drawn on the picture with the
power off kills nothing, and without them the interactive planner lies the
moment someone drags the power slider to zero.

## Labels

| quantity | p5 | median | p95 | max |
|---|---|---|---|---|
| lesion (cm²) | 4.16 | 10.24 | 28.8 | 46.72 |
| vessel (fraction of frame) | 7.8 | 13.0 | 20.9 | 28.7 |

2.9 % of instances have **no** necrosis at all — the
zero-power plans. A model scored only on cases that ablate would never be tested
on them.

## Known problems

1. **The vessel sink saturates.** The term is floored at a
   1.5 mm radius (ThermoNav's `--vessel-sink-min-radius`;
   without it 1/R² makes a venule a better heat sink than the portal vein), and
   above ~0.2 1/s it stops changing the answer — measured: rates of 2.0 and
   12.5 1/s give the same 7.48 cm² lesion. At 2 mm voxels `vessel_sink_rate`
   therefore carries information only for vessels above ~3 mm calibre. At the
   1 mm resolution of the 3-D corpus the informative range is wider.
2. **The anatomy is z-invariant.** The extrusion is what makes the label a
   deterministic function of the 2-D input, and it is also a real departure from
   a patient: no vessel enters or leaves the plane.
3. **Every lesion inherits a PROVISIONAL calibration.** See the device block.
4. **0.2 % of lesions touch the frame edge**, where the adiabatic wall
   is inside the ablation zone and the necrosis is therefore over-estimated.
   They are kept: filtering them would correlate the training set with lesion
   size, and 2 % is well below the calibration uncertainty. Do not quote a
   lesion area from a case whose zone reaches the border.
5. **33 % of lesions put more than 15 % of their area in
   connective tissue.** That is not an artefact — the placement rule requires
   the tip and the radiating slot inside the organ, not the whole zone, so a
   needle near the liver surface legitimately burns outward. It is also most of
   the reason the tissue channels carry signal.
6. **50 patients is not a population.** The splits are by patient (35/5/10),
   which is the right unit, but ten test livers is a small denominator for any
   per-case claim — read the p10 column, not only the mean.
7. **Adjacent slices of one liver are near-duplicates.** Slices are spread
   across each liver's extent rather than taken as a contiguous run, but at
   2 mm they are still correlated. The corpus is 1 600 slices, not 1 600
   independent observations.

## Generation cost

19 s median per instance single-threaded (12.76-25.85 s
p5-p95); 4000 instances on 18 pinned cores. `OMP_NUM_THREADS=1` per worker —
a 128x128x32 grid is far too small to feed 32 threads, so the
parallelism is across samples.

## Distribution

**This corpus contains real patient CT and that changes what may be shipped.**
Earlier versions were procedurally generated and carried no patient data at all;
this one is 1 600 axial slices of 50 de-identified livers from ThermoNav's
`cnca_indepth_v5_1mm`, block-pooled to 128x128 at 2 mm and windowed to the
organ. There are no headers, no identifiers and no dates in the stored arrays —
only the pixel data, a case label of the form `case-1013`, and a slice index.

What is on disk, and what is meant to travel with the repo:

| directory | what | size | ships |
|---|---|---|---|
| `data/` | the task corpus: 4 000 plans on 1 600 real slices | 41 MB | yes — the notebook loads it |
| `data_vessel/` | warm start, derived from the above | 20 MB | yes |
| `data_real/` | the extracted slices `data/` was built from | 50 MB | no — reproduction input |
| `data_vessel_wide/` | warm start, 3.3x more slices, looser criteria | 62 MB | no — too big, and `data_vessel` is the documented recipe |

That is de-identified imaging, not synthetic data. Whether it may be
redistributed inside a public tutorial repository is a governance question for
the data's owner and is **not settled by this file**. Until it is, treat the
corpus as internal: the notebook loads it from the repo, so publishing the repo
publishes the corpus.

If the answer is no, the procedural generator is still in
`ablation2d/anatomy.py` and still works — with the documented cost that the
vessel head cannot be trained on it (see the anatomy note above).
