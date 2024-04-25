import os
import pathlib


_ROOT = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent
DIR_AUDIO = _ROOT/'audio'
DIR_OUTPUTS = _ROOT/'outputs'
DIR_PRETRAINED = _ROOT/'pretrained'
