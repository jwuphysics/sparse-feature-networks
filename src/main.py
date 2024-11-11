from config import *
from trainer import *


def main():
    for target in ["metallicity", "bpt_lines"]:
        for k in [2, 3, 4, 6, 8]:

            data_config = DataConfig()
            training_config = TrainingConfig()

            experiment_config = ExperimentConfig(
                name="sparse-feature-net",
                target=target,
                k=k,
                model_dir=Path("../model"),
                results_dir=Path("../results"),
                data_config=data_config,
                training_config=training_config,
            )

            trainer = ModelTrainer(experiment_config)
            trainer.train_model()


if __name__ == "__main__":
    main()
