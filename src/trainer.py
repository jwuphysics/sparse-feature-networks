from fastai.vision.all import *
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import matplotlib.gridspec as gridspec
import cmasher as cmr
import platform

from dataloader import prepare_data, get_data_loaders
from model import ResNet18TopK
from config import ExperimentConfig


def RMSE(p, y):
    return torch.sqrt(MSELossFlat()(p, y))


class ModelTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}

        # set default plotting style
        if platform.system() == "Darwin":
            import matplotlib.font_manager as fm

            fm.fontManager.addfont("/Users/john/Library/Fonts/Nunito-Regular.otf")
            fm.fontManager.addfont("/Users/john/Library/Fonts/Nunito-Bold.otf")
            fm.fontManager.addfont("/Users/john/Library/Fonts/Nunito-ExtraBold.otf")
            plt.rcParams["font.family"] = "Nunito"
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["font.weight"] = 700
            plt.rcParams["axes.prop_cycle"] = plt.cycler(
                color=["#003f5c", "#7a5195", "#ef5675", "#ffa600"]
            )

    def train_model(self, k: int):
        """Train a model with specific k value."""
        print(f"Training model with k={k}")

        # prepare data, model, and learner
        df = prepare_data(self.config)
        dls = get_data_loaders(df, self.config)

        n_out = 1 if self.config.target == "metallicity" else 4
        model = ResNet18TopK(k=k, n_out=n_out, pretrained=True)

        learn = Learner(
            dls,
            model,
            loss_func=RMSE,
            opt_func=ranger,
        )

        model_path = (
            self.config.model_dir / f"resnet18-topk_{k}-{self.config.target}.pth"
        )
        if not model_path.exists():
            # train and save model
            learn.fit_one_cycle(
                self.config.training_config.epochs,
                self.config.training_config.learning_rate,
            )

            torch.save(learn.model, model_path)
        else:
            learn.model = torch.load(
                model_path,
                map_location=self.config.training_config.device,
                weights_only=False,
            )

        # analyze activations and create visualizations
        self.analyze_activations(learn.model, dls, k)

    @staticmethod
    def get_all_activations(loader, model):
        """Extract activations from the model."""
        activations = []
        with torch.no_grad():
            layers = nn.Sequential(*list(model.resnet.children())[:-1], nn.Flatten())
            for xb, _ in tqdm(loader):
                activations.append(layers(xb))
        return torch.concat(activations, 0).cpu().numpy()

    @staticmethod
    def create_feature_dictionary(activations):
        """Create dictionary of activation patterns."""
        feature_dict = defaultdict(list)
        for img_idx, img_activations in enumerate(activations):
            non_zero = np.nonzero(img_activations)[0]
            for feature_idx in non_zero:
                activation_strength = img_activations[feature_idx]
                feature_dict[int(feature_idx)].append(
                    (int(img_idx), float(activation_strength))
                )

        for feature_idx in feature_dict:
            feature_dict[feature_idx].sort(key=lambda x: x[1], reverse=True)

        return feature_dict

    def valid_idx_to_objid(self, idx: int, dls) -> str:
        """Convert validation index to object ID."""
        return dls.valid.items.iloc[idx].objID

    def plot_bpt_with_examples(
        self,
        k: int,
        feature_dict: dict,
        activations: np.ndarray,
        dls,
        save_dir: Path,
        min_galaxies: int = 100,
    ) -> None:
        """Create BPT diagram with example galaxy images."""
        for feat_idx in feature_dict:
            # skip zctivations with too few galaxies
            if len(feature_dict[feat_idx]) < min_galaxies:
                continue

            fig = plt.figure(figsize=(8, 4), dpi=300)
            gs = gridspec.GridSpec(3, 6, left=0.09, right=0.975, bottom=0.125, top=0.99)

            ax0 = fig.add_subplot(gs[:, :3])

            n2_ha = dls.valid.items.log_N2 - dls.valid.items.log_Ha
            o3_hb = dls.valid.items.log_O3 - dls.valid.items.log_Hb

            # get and sort by activation strength
            act_strength = activations[:, feat_idx] / activations[:, feat_idx].max()

            sort_idx = np.argsort(act_strength)
            n2_ha = n2_ha.iloc[sort_idx]
            o3_hb = o3_hb.iloc[sort_idx]
            act_strength = act_strength[sort_idx]

            # scatter plot on left
            ax0.scatter(
                n2_ha,
                o3_hb,
                c=act_strength,
                edgecolors="none",
                cmap=cmr.ember,
                s=2,
                vmin=0,
                vmax=1,
                rasterized=True,
            )

            ax0.set_xlabel("log([NII]/H$\\alpha$)", fontsize=12, fontweight="bold")
            ax0.set_ylabel("log([OIII]/H$\\beta$)", fontsize=12, fontweight="bold")
            ax0.grid(alpha=0.15)
            ax0.set_xlim(-1.55, 0.55)
            ax0.set_ylim(-1.05, 1.3)

            # plot 3x3 grid of examples on right
            for i in range(3):
                for j in range(3):
                    ax = fig.add_subplot(gs[i, j + 3])
                    image = Image.open(
                        self.config.data_config.root
                        / "data/images-sdss"
                        / f"{self.valid_idx_to_objid(feature_dict[feat_idx][i*3+j][0], dls)}.jpg"
                    )
                    ax.imshow(image, origin="lower")
                    ax.axis("off")

            fig.text(
                0.75,
                0.035,
                f"Examples of Activation {feat_idx}",
                ha="center",
                va="center",
                transform=fig.transFigure,
                fontsize=12,
                fontweight="bold",
            )

            plt.savefig(
                save_dir / f"figures/activation_{feat_idx}_bpt_scatter_examples.pdf"
            )
            plt.close()

    def plot_correlation_matrix(
        self,
        activations: np.ndarray,
        feature_dict: dict,
        save_dir: Path,
        min_galaxies: int = 100,
    ) -> None:
        """Plot correlation matrix of activations."""
        # Select features with enough galaxies
        non_zero_counts = np.sum(activations != 0, axis=0)
        normalized_activations = (activations.T / np.linalg.norm(activations, axis=1)).T
        selected_activations = normalized_activations[:, non_zero_counts > min_galaxies]

        activation_indices = np.argwhere(non_zero_counts > min_galaxies).flatten()

        if len(activation_indices) > 0:
            # compute and plot correlation matrix
            correlation_matrix = np.corrcoef(selected_activations.T)

            plt.figure(figsize=(8, 7), dpi=300)
            im = plt.imshow(
                correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1
            )

            # Add text annotations
            for i in range(correlation_matrix.shape[0]):
                for j in range(correlation_matrix.shape[1]):
                    text = plt.text(
                        j,
                        i,
                        f"{correlation_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=12,
                    )
                    if abs(correlation_matrix[i, j]) > 0.5:
                        text.set_color("white")

            plt.colorbar(label="Correlation Coefficient")
            plt.title(
                "Correlation Matrix of Selected Activations",
                fontsize=12,
                fontweight="bold",
            )
            plt.xticks(range(len(activation_indices)), activation_indices)
            plt.yticks(range(len(activation_indices)), activation_indices)

            plt.tight_layout()
            plt.savefig(save_dir / "figures/activation_correlation_matrix.pdf")
            plt.close()

    def plot_max_activating_galaxies(
        self,
        feature_dict: dict,
        activation_index: int,
        dls,
        save_dir: Path,
        top_n: int = 10,
    ) -> None:
        """
        Plot the galaxies that most strongly activate a given feature.

        Args:
            feature_dict: maps feature indices to (galaxy_idx, activation) pairs
            activation_index: index of the feature to plot
            dls: FastAI dataLoaders object
            save_dir: directory to save the plot
            top_n: number of example galaxies to show
        """
        galaxy_indices_and_activations = feature_dict[activation_index]
        top_n = min(top_n, len(galaxy_indices_and_activations))

        fig, axes = plt.subplots(1, top_n, figsize=(top_n * 1.5, 2), dpi=100, squeeze=0)
        axes = axes.reshape(-1)

        for ax, [galaxy_index, feature_activation] in zip(
            axes, galaxy_indices_and_activations[:top_n]
        ):
            image = Image.open(
                self.config.data_config.root
                / "data/images-sdss"
                / f"{self.valid_idx_to_objid(galaxy_index, dls)}.jpg"
            )
            ax.imshow(image, origin="lower")
            ax.set_title(f"{feature_activation:.4f}", fontsize=10)
            ax.axis("off")

        fig.suptitle(
            f"Activation {activation_index} ({len(galaxy_indices_and_activations)} galaxies)",
            fontsize=12,
        )
        fig.subplots_adjust(left=0, right=1, top=0.8, wspace=0.02)

        return fig

    def create_activation_visualizations(
        self, feature_dict: dict, dls, save_dir: Path
    ) -> None:
        """
        Create visualizations for all features with their maximally activating galaxies.

        Args:
            feature_dict: dict of feature activations
            dls: FastAI dataLoaders
            save_dir: directory to save visualizations
        """
        figures_dir = save_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        for k in tqdm(feature_dict, desc="Creating activation visualizations"):
            fig = self.plot_max_activating_galaxies(
                feature_dict, k, dls, figures_dir, top_n=10
            )
            fig.savefig(figures_dir / f"activation_{k}_examples.png")
            plt.close(fig)

    def get_weights(self, model, activation_index: int) -> np.ndarray:
        """Extract weights for a given activation index."""
        return model.resnet.fc.weight[:, activation_index].cpu().detach().numpy()

    def save_weights_table(
        self, model, feature_dict: dict, save_dir: Path, min_galaxies: int = 5000
    ) -> None:
        """
        Create and save a table of weights for each activation.

        Args:
            model: trained model
            feature_dict: dictionary of feature activations
            save_dir: directory to save the weights table
            min_galaxies: minimum number of galaxies for including a feature
        """
        # Get active features
        activation_indices = [
            k for k, v in feature_dict.items() if len(v) >= min_galaxies
        ]

        if not activation_indices:
            return

        # set up target features based on experiment type
        if self.config.target == "bpt_lines":
            features = ["log_N2", "log_Ha", "log_O3", "log_Hb"]
        else:  # metallicity
            features = ["oh_p50"]

        # create and fill weights array
        weights = np.zeros((len(activation_indices), len(features)))
        for j, activation_idx in enumerate(activation_indices):
            weights[j] = self.get_weights(model, activation_idx)

        # Create and save DataFrame
        weight_table = pd.DataFrame(weights, columns=features, index=activation_indices)
        weight_table.to_csv(save_dir / "weights_table.csv")

    def save_feature_stats(self, feature_dict: dict, save_dir: Path) -> None:
        """Save feature activation statistics."""
        stats = [(k, len(v)) for k, v in feature_dict.items()]
        with open(save_dir / "feature_stats.txt", "w") as f:
            f.write("Feature Index | Number of Active Galaxies\n")
            f.write("-" * 40 + "\n")
            for feat_idx, count in stats:
                f.write(f"{feat_idx:>12} | {count:>22}\n")

    def analyze_activations(self, model, dls, k):
        """Analyze and save activation patterns."""
        activations = self.get_all_activations(dls.valid, model)

        # save activations
        save_dir = self.config.results_dir / f"resnet18-topk_{k}-{self.config.target}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "figures").mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "activations.npy", activations)

        # gets feature activation strength for each galaxy
        feature_dict = self.create_feature_dictionary(activations)

        # all results
        self.create_activation_visualizations(feature_dict, dls, save_dir)
        self.save_feature_stats(feature_dict, save_dir)
        if self.config.target == "bpt_lines":
            self.plot_bpt_with_examples(k, feature_dict, activations, dls, save_dir)
        else:  # only if single target like metallicity
            self.plot_correlation_matrix(activations, feature_dict, save_dir)
            self.save_weights_table(model, feature_dict, save_dir)
