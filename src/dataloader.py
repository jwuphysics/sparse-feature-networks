from fastai.vision.all import *
from config import ExperimentConfig


def prepare_data(config: ExperimentConfig) -> pd.DataFrame:
    """Load and preprocess the SDSS galaxies data."""
    df = pd.read_csv(
        config.data_config.root / "data/galaxies.csv", dtype={"objID": str}
    )

    if config.target == "metallicity":
        df = df[(df.oh_p50 > 0) & (df.lgm_tot_p50 > 0) & (df.sfr_tot_p50 > -10)].copy()
    elif config.target == "bpt_lines":
        df = df[
            (df.nii_6584_flux / df.nii_6584_flux_err > 3)
            & (df.h_alpha_flux / df.h_alpha_flux_err > 3)
            & (df.oiii_5007_flux / df.oiii_5007_flux_err > 3)
            & (df.h_beta_flux / df.h_beta_flux_err > 3)
            & (df.nii_6584_flux < 1e5)
            & (df.h_alpha_flux < 1e5)
            & (df.oiii_5007_flux < 1e5)
            & (df.h_beta_flux < 1e5)
        ].copy()

        # compute BPT log line fluxes
        df["log_N2"] = np.log10(df.nii_6584_flux)
        df["log_Ha"] = np.log10(df.h_alpha_flux)
        df["log_O3"] = np.log10(df.oiii_5007_flux)
        df["log_Hb"] = np.log10(df.h_beta_flux)
    else:
        raise ValueError("Invalid target")

    return df


def get_data_loaders(df: pd.DataFrame, config: ExperimentConfig):
    """Create Fastai data loaders for training."""
    if config.target == "metallicity":
        y_cols = ["oh_p50"]
    else:  # bpt_lines
        y_cols = ["log_N2", "log_Ha", "log_O3", "log_Hb"]

    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_x=ColReader(
            "objID", pref=f"{config.data_config.root}/data/images-sdss/", suff=".jpg"
        ),
        get_y=ColReader(y_cols),
        splitter=RandomSplitter(1 - config.data_config.train_split, seed=config.seed),
        item_tfms=[
            Resize(config.data_config.image_size),
            CropPad(config.data_config.crop_size),
        ],
        batch_tfms=aug_transforms(
            do_flip=True,
            flip_vert=True,
            max_rotate=0,
            max_zoom=1.0,
            max_warp=0,
            p_lighting=0,
        )
        + [Normalize()],
    )

    return ImageDataLoaders.from_dblock(
        dblock, df, bs=config.training_config.batch_size
    )
