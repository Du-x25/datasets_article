# Local comparison figure verification, 2026-10-15

- Scope: existing TIFF images and saved prediction arrays only. No training, model loading, inference, linear-map reconstruction, or result rescoring.
- Main figures: the A window from C1, C4, C7, C10, selected as every third numerical site from the existing 20-window review. This is an illustrative subset, not a random sample, quality-stratified sample, or performance ranking.
- All four main windows contain 65,536 valid support pixels (100%). Supplementary C1-B contains 65,533 and C5-B contains 65,535; the remaining windows contain 65,536. The JSON and CSV record exact coordinates and fractions.
- Display: source uint8 DN values retained. Saved float32 predictions are quantized with round(255*x) for display, without per-panel normalization, gamma, sharpening, or smoothing. Excluded pixels use the same neutral gray in every panel. No spatial resampling is applied to the source crops.
- Each crop contains 256 by 256 output-grid pixels. Bars are 64 output-grid pixels, not physical distances or independent geometry measurements.
- Identity: 165 full source hashes were checked, including 60 source TIFF images, 60 saved prediction arrays, 30 result records, 10 support masks, four identity/provenance files, and the current generator. All prediction crop hashes match the earlier visual review; all result configurations use seed 11 and the matching held-out fold, dataset hash and support hash.
- Export inspection: both main PDF figures are one page and the supplementary PDF is ten pages. Text is selectable. Embedded image dimensions are 256 by 256. Main figures were inspected from both PNG output and PDF renders; all ten supplementary page previews were inspected as a contact sheet. No overlapping labels, cropped panels, or missing images were observed.
- Interpretation limit: local color and detail differences can be described for these exact windows. These plates do not establish error prevalence, model superiority, physical spectral calibration, independent alignment accuracy, or the cause of observed artifacts.

The generator, coordinate CSV and detailed provenance JSON accompany these figures.
