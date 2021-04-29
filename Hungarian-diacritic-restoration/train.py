import hydra
from omegaconf import DictConfig
from experiment import Experiment

@hydra.main(config_name="config")
def main(cfg: DictConfig):
    with Experiment(cfg) as e:
        e.run(cfg)


if __name__ == '__main__':
    main()