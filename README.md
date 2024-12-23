# Insights on Galaxy Evolution from Interpretable Sparse Feature Networks (SFNets)

We introduce sparse feature networks (SFNets), which contain a simple top-k sparsity constraint in their penultimate layers. We show that these SFNets can predict galaxy properties, such as gas metallicity or BPT line ratios, directly from image cutouts. SFNets produce interpretable feature activations, which can then be studied to better understand galaxy formation and evolution.

## Requirements

This software uses [`fastai`](https://github.com/fastai/fastai), built atop `pytorch`, and a few other packages that are commonly found in the data science stack. We've tested that this code works using `fastai==2.7.17` and `torch==2.4.1` on both Linux and macOS.

Install requirements with:
```bash
pip install torch fastai numpy pandas matplotlib cmasher tqdm
```

## Directory Structure

```
./
├── data/
│   ├── images-sdss/
│   └── galaxies.csv
├── model/
├── results/
└── src/
    ├── config.py          
    ├── dataloader.py     
    ├── model.py         
    ├── main.py             
    └── trainer.py         
```

## Usage

1. Prepare your data:
   - Construct `galaxies.csv` with the required columns (`objID`, `oh_p50` for metallicity, or line flux measurements for BPT analysis). We used CASJobs to download galaxies using [this query](https://github.com/cherryquinnlg/agn-convnets/blob/main/data/AGN_K03.sql), and then enforced a signal-to-noise ratio (SNR) cut of 3 for all spectral lines.
   - Download SDSS galaxy images into `data/images-sdss/`. We used the DESI Legacy Viewer to download via the RESTful interface, e.g. `http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&pixscale=0.262&layer=sdss&size=160`.
   

2. Run experiments: 
   - Modify and run the main `python main.py`
```python
from config import ExperimentConfig, DataConfig, TrainingConfig
from trainer import ModelTrainer

config = ExperimentConfig(
    name="metallicity_experiments",
    target="metallicity",
    k=2,
    model_dir=Path("../model"),
    results_dir=Path("../results"),
    data_config=DataConfig(),
    training_config=TrainingConfig()
)

# Train models
trainer = ModelTrainer(config)
trainer.train_model()
```

## Models and results

We have uploaded our trained model weights and sparse activation results [here](https://www.dropbox.com/scl/fo/8v0wd2r97251c4gb69iax/AAaRdE7QPFSgFOuOtsnGEEE?rlkey=14jj9mt6evtdgcqsx8ogccim9&st=zctk6mzo&dl=0). The optimized `ResNetTopK18` models should be able to reproduce the results shown in the paper.

## Citation

This paper has been submitted to AAS journals and will soon appear on arXiv. For now, please use the following citation:

```latex
@misc{
    author={Wu, John. F.},
    title={Insights on Galaxy Evolution from Interpretable Sparse Feature Networks},
    year={2024},
    howpublished={Submitted to AAS Journals}
}
```

## License

This project is licensed under the MIT License; please see the `LICENSE` file for details.