# Preprocessing: SVS → GigaTIME tiles

Converts whole-slide images (`.svs`) into 556×556 tissue tiles in the same layout as `sample_test_data`, so the existing GigaTIME dataloader and model can load them without code changes.

## Dependencies

- **OpenSlide**: `openslide-python` is in `environment.yml`. You also need the OpenSlide C library:
  - Windows: [openslide.org](https://openslide.org/) or `conda install -c conda-forge openslide`
  - Linux/macOS: package manager or conda-forge

## Usage

```python
from pathlib import Path
from preprocessing.run import process_slide, process_slides, load_config

config = load_config()  # preprocessing/config.yaml
# Process one slide
n, slide_dir = process_slide(Path("data/svs data/uuid/slide.svs"), Path("data/preprocessed_tiles"), config)
# Or all SVS under a directory (writes preprocessed_metadata.csv)
results = process_slides(Path("data/svs data"), Path("data/preprocessed_tiles"), config)
```

Then point `config["metadata"]` and `config["tiling_dir"]` in the test/train scripts at the preprocessing output directory and `preprocessed_metadata.csv`.

## Test notebook

Run `scripts/preprocessing_test.ipynb` to process slides from `data/svs data/` and verify model compatibility (generate_tile_pair_df + one batch).
