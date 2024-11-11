# Learning Galaxy Astrophysics with Interpretable Sparse Feature Networks (SFNets)

We introduce sparse feature networks (SFNets), which contain a simple top-k sparsity constraint in their penultimate layers. We show that these SFNets can predict galaxy properties, such as gas metallicity or BPT line ratios, directly from image cutouts. SFNets produce interpretable feature activations, which can then be studied to better understand galaxy formation and evolution.

## Requirements

- `python>=3.12`
- `pytorch`
- `fastai>=2.0`
- `numpy`
- `pandas`
- `matplotlib`
- `cmasher`
- `tqdm`

Install requirements with:
```bash
pip install torch fastai numpy pandas matplotlib cmasher tqdm
```

## Project Structure

```
project/
├── data/
│   ├── images-sdss/       # SDSS image cutouts in JPG format
│   └── galaxies.csv       # SDSS catalog
├── model/                 # model weights
├── results/               # generated results and figures
└── src/
    ├── config.py          
    ├── dataloader.py     
    ├── model.py         
    ├── main.py             
    └── trainer.py         
```

## Usage

1. Prepare your data:
   - Place SDSS galaxy images in `data/images-sdss/`
   - Ensure `galaxies.csv` contains required columns (objID, oh_p50 for metallicity, or line flux measurements for BPT analysis)

2. Run experiments: 
   - Modify and run the main `python main.py`
```python
from config import ExperimentConfig, DataConfig, TrainingConfig
from trainer import ModelTrainer

config = ExperimentConfig(
    name="galaxy_analysis",
    target="metallicity",
    k=4,
    model_dir=Path("../model"),
    results_dir=Path("../results"),
    data_config=DataConfig(),
    training_config=TrainingConfig()
)

# Train models
trainer = ModelTrainer(config)
trainer.train_model()
```

## Citation

This paper has been submitted to AAS journals and will soon appear on arXiv. For now, please use the following citation:

```latex
@misc{
    author={Wu, John. F.},
    title={Learning Galaxy Astrophysics from Interpretable Sparse Feature Networks},
    year={2024},
    howpublished={Submitted to AAS Journals}
}
```

## License

This project is licensed under the MIT License; please see the LICENSE file for details.