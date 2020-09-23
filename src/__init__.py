from src.config import Config
from src import models, trainer, dataset, transforms, utils

config = Config().get_yaml()
utils.set_random_seeds(config.get("seed"))
