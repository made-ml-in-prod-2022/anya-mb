from omegaconf import DictConfig, OmegaConf
import hydra

# from ml_project.src.train_model import train_pipeline, TrainingParams
from train_model import train_pipeline, TrainingParams


MODELS_PATH = '../models'
DATA_PATH = '../data/heart_cleveland_upload.csv'
TARGET_COLUMN = 'condition'

RANDOM_STATE = 91
TEST_SIZE = 0.2
MODEL_NAME = 'RF'


@hydra.main(config_path='../config', config_name='config')
def app(cfg: DictConfig):
    print(cfg)

    if cfg['mode']['train']:

        test_size = cfg['mode']['test_size']
        model = cfg['model']
        random_state = cfg['general']['random_state']
        model_path = cfg['general']['model_path']
        data_path = cfg['mode']['data_path']
        target_column = cfg['general']['target_column']

        train_cfg = TrainingParams(
            model, random_state, model_path, test_size, data_path, target_column
        )

        train_pipeline(train_cfg)

    else:
        pass


if __name__ == '__main__':
    app()
