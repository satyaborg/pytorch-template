import yaml
import os

class Config(object):
    def __init__(self):
        pass
    
    def get_yaml(self):
        with open("config.yaml", 'r') as stream:
            return yaml.safe_load(stream)