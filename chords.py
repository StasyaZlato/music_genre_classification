from typing import Tuple, NoReturn

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import spatial
from sklearn.preprocessing import normalize

from chords_masks import get_triads_masks
from constants import PITCH_CLASSES


class ChordsExtractor:
    def __init__(self, sr=44000, nc_coef=0.6, seventh_coef=0.7):
        self._chords_masks = get_triads_masks()
        self._sr = sr
        self._nc_sensitivity_coef = nc_coef
        self._seventh_sensitivity_coef = seventh_coef

    def _print_chord(self, mask: Tuple[float]) -> str:
        """
        gets chord name

        :param mask: chord mask with harmonics
        :return: chord name
        """
        return self._chords_masks[mask]

    def plt_chord_masks(self, cols: int, height=60, width=60) -> NoReturn:
        """
        visualize chord masks

        :param width: width of figure
        :param height: height of figure
        :param cols: number of columns in grid
        """
        chords = list(self._chords_masks.keys())
        rows = len(chords) // cols
        fig, axs = plt.subplots(rows, cols)
        fig.set_figheight(height)
        fig.set_figwidth(width)

        index = 0
        for row in range(rows):
            for col in range(cols):
                if index >= len(chords) - 1:
                    break
                axs[row, col].bar(PITCH_CLASSES, chords[index])
                axs[row, col].set_title(self._print_chord(chords[index]))
                index += 1

        plt.show()

    def process_composition_librosa(self, path: str) -> Tuple[ArrayLike[int], NDArray[float], ArrayLike[float], float]:
        """
        calculate all necessary librosa features for given composition

        :param path: path to composition
        :return:    beats - list of beats moments in composition (by time frames)
                    chroma_stft - analogue of PCP representation for composition
                    chords_time - list of beats moments in composition (by seconds)
                    duration - total duration of composition in seconds
        """
        y, sr = librosa.load(path, sr=self._sr)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chords_time = librosa.frames_to_time(beats, sr=sr)

        return beats, chroma_stft, chords_time, duration

    def get_chords_per_frame(self, chroma_stft: NDArray[float], beats: ArrayLike[int], chords_time: ArrayLike[float],
                             total_duration: float) -> Tuple[ArrayLike[str], ArrayLike[float]]:
        """
        calculates chords for each interval (from beat to beat)

        :param chroma_stft: PCP-like representation of composition with shape (12,x)
        :param beats: moments of beats (per time frame)
        :param chords_time: moments of beats (via seconds from start of composition)
        :param total_duration: duration of composition in seconds
        :return: list of chords and list of their durations
        """
        chroma_chords = np.split(chroma_stft, beats)

        chords_masks = list(self._chords_masks.keys())
        chords = []

        for chroma in chroma_chords:
            if len(chroma) == 0:
                continue
            vector = normalize([np.mean(chroma, axis=0)], axis=1, norm="max")[0]
            max_sim = 0
            probable_mask = self._print_chord(chords_masks[0])
            for mask in chords_masks:
                sim = 1 - spatial.distance.cosine(mask, vector)
                if np.all(np.array(mask) == 1):
                    sim *= self._nc_sensitivity_coef  # sensitivity tuning
                if "7" in self._print_chord(mask):
                    sim *= self._seventh_sensitivity_coef
                if sim > max_sim:
                    max_sim = sim
                    probable_mask = self._print_chord(mask)
            chords.append(probable_mask)

        times = []
        for i in range(len(chords)):
            if i == 0:
                times.append(chords_time[i])
            elif i < len(chords_time):
                times.append(chords_time[i] - chords_time[i - 1])
            else:
                times.append(total_duration - chords_time[i - 1])

        return chords, times

    def process_composition(self, path: str) -> Tuple[ArrayLike[str], ArrayLike[float]]:
        """
        the whole pipeline of processing composition from audio file

        :param path: path to composition
        :return: list of chords and list of their durations
        """
        beats, chroma_stft, chords_time, total_duration = self.process_composition_librosa(path)
        return self.get_chords_per_frame(chroma_stft.T, beats, chords_time, total_duration)

    def process_composition_from_dataset(
            self, chroma: NDArray[float], beats: ArrayLike[int], chords_time: ArrayLike[float], total_duration: float) \
            -> Tuple[ArrayLike[str], ArrayLike[float]]:
        """
        the whole pipeline of processing dataset entry with precalculated features

        :param chroma: PCP-representation of composition
        :param beats: beats moments via time frame
        :param chords_time: moments of beats (via seconds from start of composition)
        :param total_duration: duration of composition in seconds
        :return: list of chords and list of their durations
        """
        return self.get_chords_per_frame(chroma, beats, chords_time, total_duration)
