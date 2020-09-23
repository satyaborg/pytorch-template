# #%%
# %load_ext autoreload
# %autoreload 2
# to run as a Jupyter cell in VSCode add #%%
from src import config
from src.trainer import Trainer
from src.preprocessor import Preprocessor
from src.utils import *

if __name__ == "__main__":
    trainer = Trainer(**config)
    trainer.run()

