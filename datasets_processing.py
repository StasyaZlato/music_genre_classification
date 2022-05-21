import pandas as pd
import os
import re
import numpy as np
import operator


def _get_chord_from_guitarset_file(file):
    matcher = re.search(r'^(\w#?)(\w{3}).wav$', file)
    if matcher:
        return matcher.group(1) + "_" + matcher.group(2)


def guitar_set(path):
    files = sorted(os.listdir(path))

    return {os.path.join(path, file): _get_chord_from_guitarset_file(file) for file in files}


def _prepare_ccd_dataset(path):
    df = pd.read_csv(path)

    df_list = np.split(df, df[df.notnull().all(1)].index)

    result = []

    for df in df_list:
        df = df.drop(df[df.notnull().all(1)].index)
        result.append(df.iloc[:, 1:])
    return result


def _get_time_ccd(chords_time):
    times = []
    for i in range(len(chords_time)):
        if i == 0:
            times.append(chords_time[i])
        elif i < len(chords_time):
            times.append(chords_time[i] - chords_time[i - 1])
    return np.array(times)


def process_features_ccd(df_chroma, df_chords):
    chroma_features_time = (df_chroma.iloc[:, 0].values * 10).astype(int)
    chords_time = (df_chords.iloc[:, 0].values * 10).astype(int)

    beats = np.where(np.isin(chroma_features_time, chords_time))
    chroma = df_chroma.iloc[:, 1:].to_numpy()

    chords_durations = _get_time_ccd(chords_time)
    total_duration = np.sum(chords_durations)
    # в работе используется гамма C, C#, D, ..., B, а в датасете - A, A#, B, ..., G#, поэтому нужен сдвиг с
    # np.roll(ax, shift)
    return beats[0], np.roll(chroma, 9), chords_durations, total_duration


# to change pc like Bb to A# according to chosen notation
def _get_prev_pc_ccd(pc):
    chords = 'ABCDEFG'
    index = operator.indexOf(chords, pc)
    return chords[index - 1]


def _get_triad_ccd(chord):
    matcher = re.search(r"^(.*?_\w{3})_\w{3}7$", chord)
    if matcher:
        triad = matcher.group(1)
    else:
        triad = chord
    main_pc = re.search(r"^(\w)b", triad)
    if main_pc:
        return re.sub(r"^\wb", _get_prev_pc_ccd(main_pc.group(1)) + "#", triad).lower()
    return triad.lower()


def _process_chords_as_triads_ccd(df_chords):
    chords = [_get_triad_ccd(chord) for chord in df_chords.iloc[:, 1].tolist()]
    return chords


def get_chords_to_compare(df_chords):
    chords_precalculated = _process_chords_as_triads_ccd(df_chords)

    return chords_precalculated


def cross_composer_dataset(path_to_chords_csv, path_to_nnls_csv):
    chroma_bach = _prepare_ccd_dataset(path_to_nnls_csv)
    chords_bach = _prepare_ccd_dataset(path_to_chords_csv)

    return chroma_bach, chords_bach
