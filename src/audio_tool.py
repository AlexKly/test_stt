import os
import json
import torch
import librosa
import pathlib
import whisper
import soundfile as sf
from typing import Union

from src import DIR_OUTPUTS, DIR_PRETRAINED


class AudioTool:
    def __init__(self, modelname: str, device: str = 'cpu') -> None:
        """ Audio tool for modifying and transcribing audio files.

        :param modelname: Whisper model name.
        :param device: PyTorch device.
        """
        device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self._mkdir(path=DIR_PRETRAINED)
        # At first time, you need to download model:
        if modelname in [f for f in os.listdir(path=DIR_PRETRAINED) if os.path.isfile(os.path.join(DIR_PRETRAINED, f))]:
            self.model = whisper.load_model(name=DIR_PRETRAINED/modelname, device=device)
        else:
            modelname = modelname.replace('.pt', '')
            self.model = whisper.load_model(name=modelname, device=device, download_root=DIR_PRETRAINED)

    @staticmethod
    def _mkdir(path: Union[str, pathlib.Path]) -> None:
        """ Create directory if it doesn't already exist.

        :param path: Path where the directory must be created.
        :return:
        """
        if not os.path.exists(path=path):
            os.mkdir(path=path)

    def _save_report(self, report: dict) -> None:
        """ Save transcription (model result) to json file.

        :param report: Transcribed text from audio.
        :return:
        """
        self._mkdir(path=DIR_OUTPUTS)
        with open(file=DIR_OUTPUTS/'report.json', mode='w', encoding='utf-8') as f:
            json.dump(obj=report, fp=f, ensure_ascii=False)

    def modify(self, path: Union[str, pathlib.Path], volume_coef: float = None, speed_coef: float = None) -> None:
        """ Modify audio file. Two options available for modification:\n
        * bass boost - 'volume_coef' controls.
        * speed up/down - 'speed_coef' controls.

        :param path:
        :param volume_coef: Audio volume increase/decrease coefficient. Can be from 0 to 10. Default: None (ignored).
        :param speed_coef: Audio speed increase/decrease coefficient. Can be from 0 to 10. Default: None (ignored).
        :return:
        """
        samples, sr = librosa.load(path=path, sr=None)
        if volume_coef is not None:
            if volume_coef > 10:
                raise ValueError('Value too high')
            samples = samples * volume_coef
        if speed_coef is not None:
            if volume_coef > 10:
                raise ValueError('Value too high')
            samples = librosa.effects.time_stretch(y=samples, rate=speed_coef)
        self._mkdir(path=DIR_OUTPUTS)
        sf.write(file=DIR_OUTPUTS/'changed.wav', data=samples, samplerate=sr)

    def transcribe(self, path: Union[str, pathlib.Path]) -> None:
        """ Transcribe audio.

        :param path: Path to audio file.
        :return:
        """
        samples, sr = librosa.load(path=path, sr=16000)     # Sample Rate is fixed for Whisper
        result = self.model.transcribe(audio=samples, verbose=False)
        self._save_report(report={'text': result['text']})
