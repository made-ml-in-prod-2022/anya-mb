import sys
import logging

from omegaconf import DictConfig, OmegaConf
import hydra

from train import train_pipeline, TrainingParams
from predict import predict_pipeline, PredictingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(config_path='../config', config_name='config')
def app(cfg: DictConfig):
    print(cfg)

    model_path = cfg['general']['model_path']
    log_path = cfg['general']['log_path']
    target_column = cfg['general']['target_column']

    if cfg['mode']['train']:

        random_state = cfg['general']['random_state']
        model = cfg['model']
        test_size = cfg['mode']['test_size']
        data_path = cfg['mode']['data_path']

        train_cfg = TrainingParams(
            model,
            random_state,
            model_path,
            test_size,
            data_path,
            target_column,
            log_path
        )

        train_pipeline(train_cfg)

    else:
        data_path = cfg['mode']['data_path']
        predictions_path = cfg['mode']['predictions_path']

        predict_cfg = PredictingParams(
            model_path,
            data_path,
            log_path,
            predictions_path,
            target_column
        )
        predict_pipeline(predict_cfg)


if __name__ == '__main__':
    app()
