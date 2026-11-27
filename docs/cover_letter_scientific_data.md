# Cover Letter — *Scientific Data* Data Descriptor Submission

**Date:** [Insert submission date]  
**To:** The Editors, *Scientific Data*  
**Manuscript Title:** M-SWIR-RGB: Scene-Level Paired RGB and Five-Band Short-Wave Infrared Dataset for Cross-Spectral Synthesis  
**Article Type:** Data Descriptor  
**Authors:** Hu Du, Xiao Liu, Lili Du, and Hu Cui  
**Corresponding Authors:** Hu Du (du-x@mail.ustc.edu.cn) and Xiao Liu (liux@aiofm.ac.cn)

---

Dear Editors,

We are pleased to submit our Data Descriptor manuscript titled **"M-SWIR-RGB: Scene-Level Paired RGB and Five-Band Short-Wave Infrared Dataset for Cross-Spectral Synthesis"** for consideration in *Scientific Data*.

Short-wave infrared (SWIR, 900–1700 nm) imaging is increasingly deployed in ground-level, autonomous, and UAV perception systems because of its strong material reflectance contrasts and reduced atmospheric scattering. However, supervised cross-spectral SWIR-to-RGB synthesis has long lacked an open, traceable, scene-level resource with multiple narrow SWIR sub-bands registered to visible RGB on identical spatial support. Existing public datasets either offer only a single broadband SWIR channel, suffer depth-dependent baseline parallax from separate camera apertures, or release unlinked patches without source-frame provenance or site-disjoint evaluation protocols.

In this Data Descriptor, we introduce **M-SWIR-RGB**, comprising:
- **60 registered six-modality uncompressed eight-bit TIFF mosaics** (RGB plus five nominal 50 nm narrow-band SWIR sub-bands, B1–B5) and **840 deterministically selected source frames** across **10 distinct physical ground sites** (140 overlapping camera positions), acquired on a common-aperture dichroic beamsplitting platform that optically suppresses baseline parallax.
- **Complete FAIR provenance and usability support**, including SHA-256 checksum manifests (1,109 verified files), acquisition date provenance records, pipeline geometric optimization diagnostics (bundle adjustment RMS, cross-modal RGB–B1 holdout P95 residuals, and intra-SWIR sub-band P95 residuals), an independent non-SIFT gradient-domain sub-pixel phase-correlation audit across all 15 modality pairs (median 0.51 px SWIR–SWIR and 2.37 px RGB–SWIR), dynamic-range masks ($0 < \mathrm{DN} < 250$), and complete privacy redaction (seven C10 vehicle licence plates directly redacted across all six mosaics, plus frame-level privacy screening of all 840 selected source frames with all eight visible C10 vehicles redacted across RGB and B1–B5).
- **Transparent, multi-protocol technical validation** separating three distinct operating regimes: (i) Track A window-level compatibility and sub-band ablation with unmodified open-source architectures (TFD-GAN, DRUNet, pix2pix), exposing how random 4:1 window splits place 701 of 710 test windows inside the training union; (ii) Track B balanced site-disjoint five-fold cross-validation across 225 trained models plus 45 random-split controls, quantifying a +4.435 dB spatial-overlap inflation and revealing why uncalibrated cross-site colour shift limits zero-shot neural generalization; and (iii) Track C few-shot target-site colour calibration (recovering affine PSNR from 14.113 to 16.502 dB and quadratic to 16.715 dB with 10 paired target patches) compared directly against ten-patch full-parameter DRUNet fine-tuning on the identical 500,000 held-out pixels per site (+1.070 dB over zero-shot to 14.448 dB, surpassing pixel calibration on seam-heavy C9 while exposing full-network overfitting on the remaining nine sites).

**Data Availability & Repository Status:**  
All data analysis, manifest verification, release-packaging, and manuscript table/figure regeneration scripts are openly maintained under the MIT license at `https://github.com/Du-x25/datasets_article`. The complete M-SWIR-RGB dataset v1.0.0 (775 MiB across three modular ZIP archives plus top-level manifests; 1.26 GiB unpacked across 1,109 SHA-256-verified files; CC BY 4.0) is deposited in **Zenodo** with permanent DOI `https://doi.org/10.5281/zenodo.24575028` (currently held in reserved-DOI draft / [insert reviewer private access link if published as restricted for peer review], and ready for full public release upon publication). All 60 mosaics and 840 selected source frames have completed privacy screening and pixel-level licence-plate redaction prior to upload.

We confirm that this work is original, has not been published elsewhere, is not under consideration by another journal, and that all authors have approved the manuscript and declare no competing interests. We believe M-SWIR-RGB will serve as a transparent, rigorous benchmark resource for the cross-spectral imaging, multimodal vision, and remote-sensing communities.

Thank you for your time and consideration.

Sincerely,

**Hu Du** and **Xiao Liu** (on behalf of all authors)  
University of Science and Technology of China & Anhui Institute of Optics and Fine Mechanics, Hefei Institutes of Physical Science, Chinese Academy of Sciences
