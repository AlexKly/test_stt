from src import DIR_AUDIO
from src.audio_tool import AudioTool

tool = AudioTool(modelname='base.pt')
VOLUME_COEFFICIENT = 2.5
SPEED_COEFFICIENT = 0.5
FILENAME = 'example_ru.wav'


def run() -> None:
    # 1. Perform audio modification:
    tool.modify(path=DIR_AUDIO/FILENAME, volume_coef=VOLUME_COEFFICIENT, speed_coef=SPEED_COEFFICIENT)
    # 2. Transcribe audio:
    tool.transcribe(path=DIR_AUDIO/FILENAME)


if __name__ == '__main__':
    run()
