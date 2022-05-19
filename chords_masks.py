import numpy as np
from sklearn.preprocessing import normalize

from constants import HARMONICS_COEF, PITCH_CLASSES, SEVENTH


def get_harmonics(mask):
    """
    Defines the harmonics that sound simultaneously with the fundamental notes of chord.
    For a pitch class harmonic is a pc with frequency that is multiple of the fundamental, but with lower intensity.
    The first harmonics can be found by adding the intervals to the fundamental one by one:
    h1 = n + octave
    h2 = h1 + perfect fifth
    h3 = h2 + perfect fourth
    h4 = h3 + major third
    ... and so on.
    As harmonics has less intensity than fundamentals, their weights in the mask should be less than 1
    (in contrary to fundamentals that originally form binary mask like [0,1,0,0,0,1,0,0,1...]).
    We use function from Oudre L., Grenier Y., F ́evotte C. 2011 for calculating harmonics weights:
    c^(k-1) where c is empirically chosen coefficient and k is order of harmonic (here c = 0.5)

    :param mask: binary mask of length 12 corresponding to 12 pitch classes
    :return: list of 12 lists with found harmonics for each pitch class (i.e. harmonics that correspond to the given pc)
    for ex. [[], [1, h2_w, h3_w], [], ...], which means that fundamental note, 2nd harmonic of some pc
    and 3d harmonic of some (possibly) other pc sound in c#
    """
    harmonics = [[1] if el == 1 else [] for el in mask]
    for i in range(len(harmonics)):
        if 1 in harmonics[i]:
            perfect_fifth = (i + 7) % 12
            perfect_fourth = (perfect_fifth + 5) % 12
            major_third = (perfect_fourth + 4) % 12

            harmonics[perfect_fifth].append(2)
            harmonics[perfect_fourth].append(3)
            harmonics[major_third].append(4)
    return harmonics


def get_weights(harmonic_array):
    """
    calculates the total weight for the given pitch class according to harmonics

    :param harmonic_array: list of weights of sounding harmonics weights
    :return: the weight of the pc in total for mask
    """
    np_harmonic_ar_m1 = np.array(harmonic_array) - 1
    return np.sum(HARMONICS_COEF ** np_harmonic_ar_m1)


def get_perfect_fifth_mask(main_pc_index, is_major):
    """
    generates the binary mask for a triad chord with given basis

    :param main_pc_index: the basis of the chord (it"s id in PITCH_CLASSES list)
    :param is_major: true if the generated chord is major
    :return: the binary mask for chord
    """
    if main_pc_index >= 12:
        print("error")
        return

    if is_major:
        diff = 4
    else:
        diff = 3

    fifth = 7
    mask = [0 for _ in range(12)]
    mask[main_pc_index] = 1

    mask[(main_pc_index + fifth) % 12] = 1
    mask[(main_pc_index + diff) % 12] = 1
    return mask, PITCH_CLASSES[main_pc_index] + ("_maj" if is_major else "_min")


def get_seventh_mask(index):
    """
    generates binary masks for the seventh chords with given basis

    :param index: index of base pc
    :return: list of masks for min, maj, dom, dim, half_dim seventh chords
    """
    if index >= 12:
        print("error")
        return

    masks = {}
    for key, val in SEVENTH.items():
        mask = [0 for _ in range(12)]

        mask[index] = 1

        for diff in val:
            mask[(index + diff) % 12] = 1

        masks[PITCH_CLASSES[index] + key] = mask
    return masks


def get_mask_with_harmonics(mask):
    """
    updates a binary masks with harmonics

    :param mask: binary mask for chord
    :return: normalized mask with harmonics
    """
    harmonics = get_harmonics(mask)
    pc_weights = np.array([get_weights(harmonic) for harmonic in harmonics])

    return normalize([pc_weights], norm="max", axis=1)[0]


def get_triads_masks():
    """
    generates all major and minor triads masks

    :return: mapping mask (with harmonics) to chord_name
    """
    chords_reverted = {}
    for index in range(12):
        min_triad, name_1 = get_perfect_fifth_mask(index, True)
        maj_triad, name_2 = get_perfect_fifth_mask(index, False)

        min_triad_h = get_mask_with_harmonics(min_triad)
        maj_triad_h = get_mask_with_harmonics(maj_triad)

        chords_reverted[tuple(min_triad_h.tolist())] = name_1
        chords_reverted[tuple(maj_triad_h.tolist())] = name_2

    no_chord = [1] * 12
    chords_reverted[tuple(no_chord)] = "N"

    return chords_reverted


def get_triads_base_masks():
    """
    generates all major and minor triads binary masks

    :return: mapping chord_name to binary_mask
    """
    base_masks = {}
    for index in range(12):
        min_triad, name_1 = get_perfect_fifth_mask(index, True)
        maj_triad, name_2 = get_perfect_fifth_mask(index, False)

        base_masks[name_1] = min_triad
        base_masks[name_2] = maj_triad

    no_chord = [1] * 12
    base_masks["N"] = no_chord

    return base_masks
