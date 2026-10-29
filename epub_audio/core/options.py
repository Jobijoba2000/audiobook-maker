"""Processing options independent of the console and external tools."""
from dataclasses import dataclass
import math
from epub_audio.config.defaults import (DEFAULT_SPEED, DEFAULT_BITRATE, DEFAULT_WORKERS,
                                        PARAGRAPH_PAUSE_MS, SCENE_PAUSE_MS, DEFAULT_SAMPLE_RATE,
                                        DEFAULT_AUDIO_FORMAT, FORMAT_BITRATES, FORMAT_SAMPLE_RATES,
                                        SEGMENTATION_MODES, DEFAULT_SEGMENTATION, DEFAULT_CONVERT_NUMBERS,
                                        DEFAULT_NORMALIZE)


@dataclass(frozen=True)
class AudioOptions:
    speed: float = DEFAULT_SPEED
    bitrate: int | None = DEFAULT_BITRATE
    workers: int = DEFAULT_WORKERS
    paragraph_pause_ms: int = PARAGRAPH_PAUSE_MS
    scene_pause_ms: int = SCENE_PAUSE_MS
    audio_format: str = DEFAULT_AUDIO_FORMAT
    sample_rate: int | None = None
    segmentation: str = DEFAULT_SEGMENTATION
    convert_numbers: bool = DEFAULT_CONVERT_NUMBERS
    normalize: bool = DEFAULT_NORMALIZE

    def __post_init__(self):
        if type(self.normalize) is not bool:
            raise ValueError('Normalisation invalide : utilisez true ou false.')
        if type(self.convert_numbers) is not bool:
            raise ValueError('Conversion des nombres invalide : utilisez true ou false.')
        if not isinstance(self.segmentation, str) or self.segmentation not in SEGMENTATION_MODES:
            raise ValueError('Découpage des segments inconnu : ' + str(self.segmentation))
        if not math.isfinite(self.speed) or not 0.5 <= self.speed <= 3:
            raise ValueError('La vitesse doit être comprise entre 0.5 et 3.')
        if self.audio_format not in FORMAT_BITRATES:
            raise ValueError('Format audio inconnu : ' + str(self.audio_format))
        if self.sample_rate is None:
            object.__setattr__(self, 'sample_rate', 24000 if self.audio_format == 'wav' else DEFAULT_SAMPLE_RATE)
        if self.audio_format == 'wav':
            object.__setattr__(self, 'bitrate', None)
        if self.audio_format != 'wav' and (type(self.bitrate) is not int
                or self.bitrate not in FORMAT_BITRATES[self.audio_format]):
            raise ValueError('Débit audio non pris en charge : ' + str(self.bitrate))
        if type(self.sample_rate) is not int or self.sample_rate not in FORMAT_SAMPLE_RATES[self.audio_format]:
            raise ValueError(f'Fréquence {self.sample_rate} Hz incompatible avec {self.audio_format.upper()}.'
                             + (' Opus accepte 24 ou 48 kHz.' if self.audio_format == 'opus' else ''))
        if not 1 <= self.workers <= 20:
            raise ValueError('Le nombre de workers doit être compris entre 1 et 20.')
        if self.paragraph_pause_ms < 0 or self.scene_pause_ms < 0:
            raise ValueError('Les pauses doivent être positives ou nulles.')
