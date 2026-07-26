# Figure contracts

## Figure 1 — scene and modality overview

- Core conclusion: the release exposes ten paired scene units with one RGB and five SWIR views per scene.
- Archetype: image plate.
- Hero evidence: directly sampled pixels from all sixty released scene-level TIFF files.
- Source data: dataset file manifest and the released TIFF files.
- Integrity: each complete image receives one global 1st–99th percentile display stretch; no local editing or content-aware replacement.
- Reviewer risk: thumbnails demonstrate organization and visual diversity, not registration accuracy.
- Exports: SVG, PDF, TIFF at 300 dpi, and PNG preview.

## Figure 2 — scale and quality diagnostics

- Core conclusion: the dataset combines heterogeneous scene extents with a fully auditable patch inventory and registration diagnostics.
- Archetype: quantitative grid with a dominant patch-count panel.
- Panels: usable patch count; scene dimensions; RGB–B1 held-out residual; RGB–B1 corner deviation.
- Statistics: descriptive scene-level values; no inferential test.
- Source data: exported scene audit and scene quality summary.
- Reviewer risk: automated feature residuals must not be labelled manual-checkpoint RMSE.
- Exports: SVG, PDF, TIFF at 600 dpi, and PNG preview.

## Figure 3 — data flow and project boundary

- Core conclusion: immutable source observations flow through explicit processing and quality gates into scene-level data, patches, experiment manifests, and manuscript evidence.
- Archetype: schematic-led workflow.
- Evidence: repository contracts and validated file relationships.
- Reviewer risk: do not depict a processing operation that is absent from code or provenance.
- Exports: SVG, PDF, TIFF at 600 dpi, and PNG preview.
