from collections import Counter
from itertools import combinations, permutations
from typing import List, Dict, Union, Tuple, NoReturn, Hashable, Sequence

import matplotlib.pyplot as plt
from pylab import MaxNLocator
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from simplicial import Filtration
from persim import PersistenceImager
import seaborn as sns

from chords_masks import get_triads_base_masks
from constants import PITCH_CLASSES
from utils import get_combinations, get_gauss_form, get_low, normalize_dict_values

Point = Tuple[int, int]
VerticesDict = Dict[int, Point]


class Tonnetz:
    def __init__(self, tonnetz_type="trajectory", filtration_step=0.005, pixel_size=0.2):
        self._base_masks = get_triads_base_masks()
        self._tonnetz_type = tonnetz_type
        self._filtration_step = filtration_step
        if tonnetz_type == "trajectory":
            self._tonnetz = [4, 3, 5]
            self._tonnetz_axis = {"x": self._tonnetz[0], "y": self._tonnetz[1], "xy": self._tonnetz[2]}
            self._initial_grid = {i: (i % 3, -i % 4) for i in range(12)}

        self._trajectory = None
        self._persistence_diagram = None
        self._filtration = None
        self._pers_img = None

        self._persimg = PersistenceImager(pixel_size=pixel_size)

    def build_filtration(self, chords: ArrayLike, durations: ArrayLike) -> Filtration:
        """
        build simplicial filtration on chords sequence

        :param chords: list of chords
        :param durations: list of their durations
        :return: Filtration
        """
        if self._tonnetz_type == "frequency":
            self._filtration = self._build_tonnetz_filtration_on_frequencies(chords)
            return self._filtration
        if self._tonnetz_type == "trajectory":
            assert durations is not None, "Duration list can not be None"
            self._filtration = self._build_filtration_on_tonnetz_trajectory(chords, durations)
            return self._filtration

    def _get_filtration_level(self, a_max: float, a: float) -> int:
        """
        computes the level of filtration using equation for arithmetic progression with d = filtration_step as we
        apply that the simplices with the longest duration have the minimal distance, i.e. should appear in
        filtration before simplices with smaller duration, we consider -d with and go from a_max (equal to max el) to 0

        :param a_max: maximum weight among simplices
        :param a: current weight among simplices
        :return: on which step of filtration the simplex appears
        """
        a1 = a_max // self._filtration_step * self._filtration_step
        an = a // self._filtration_step * self._filtration_step

        return round((an - a1) / (-self._filtration_step) + 1) # an = (n-1)*step + a1

    def _get_filtration(self, weighted_simplices_dict: Dict[Hashable, float]) -> Dict[int, List[Hashable]]:
        """
        defines steps of the filtration for the set of weighted simplices

        :param weighted_simplices_dict: dict {simplex : weight}, where weight is either frequency or normalized by
        max norm duration
        :return: dict with mapping {step of filtration : list of simplices appearing on step}  
        """
        max_el = max(weighted_simplices_dict.values())
        filtration_dict = {}
        for key, val in weighted_simplices_dict.items():
            filtration_step = self._get_filtration_level(max_el, val)
            if filtration_step in filtration_dict:
                filtration_dict[filtration_step].append(key)
            else:
                filtration_dict[filtration_step] = [key]
        return filtration_dict

    def _get_pc_indexes(self, chord: str) -> ArrayLike:
        """
        get numerical representation of pitch classes that appear in chord
        :param chord: chord name (str)
        :return: indexes of the pitch classes
        """
        mask = self._base_masks[chord]
        return np.where(np.array(mask) == 1)[0]

    # ---------- Tonnetz based on frequencies ----------

    def _get_simplices(self, chord: str) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """
        generates all simplices for frequency-tonnetz representation. Each pc, interval and chord appears exactly ones
        
        :param chord: chord name 
        :return: simplices of 0d (pitch classes), 1d (intervals) and 2d (triads) as 3 lists of tuples
        """
        pitch_classes = np.array(PITCH_CLASSES)[self._get_pc_indexes(chord)]

        combs = get_combinations(pitch_classes)
        intervals = [tuple(sorted(x)) for x in combs if len(x) == 2]
        triads = [tuple(sorted(x)) for x in combs if len(x) == 3]

        return [(x,) for x in pitch_classes], intervals, triads

    def _get_frequencies_dict(self, chords: List[str]) -> Dict[Tuple, float]:
        """
        calculates frequency dict for all simplices
        all frequencies are calculated with respect to number of chords
        
        :param chords: sequence of chords 
        :return: dict {simplex : frequency}
        """
        chords_cnt = dict(Counter(chords))

        pc_freq_dict = {}
        intervals_freq_dict = {}
        triads_freq_dict = {}

        for chord in chords:
            pitch_classes, intervals, triads = self._get_simplices(chord)
            cnt = chords_cnt[chord]
            for pc in pitch_classes:
                pc_freq_dict[pc] = pc_freq_dict.get(pc, 0) + cnt
            for interval in intervals:
                intervals_freq_dict[interval] = intervals_freq_dict.get(interval, 0) + cnt
            for triad in triads:
                triads_freq_dict[triad] = triads_freq_dict.get(triad, 0) + cnt

        total_triads_cnt = sum(triads_freq_dict.values())

        pc_freq_dict = {key: value / total_triads_cnt for key, value in pc_freq_dict.items()}
        intervals_freq_dict = {key: value / total_triads_cnt for key, value in intervals_freq_dict.items()}
        triads_freq_dict = {key: value / total_triads_cnt for key, value in triads_freq_dict.items()}

        freq_dict = {}
        freq_dict.update(pc_freq_dict)
        freq_dict.update(intervals_freq_dict)
        freq_dict.update(triads_freq_dict)

        return freq_dict

    def _build_tonnetz_filtration_on_frequencies(self, chords: List[str]) -> Filtration:
        """
        builds Filtration based on chords frequencies
        the resulting max simplicial complex has pitch classes as vertices, intervals as edges and chords as triangles,
        each simplex appears only once, so the representation is quite compact

        :param chords: sequence of chords
        :return: Filtration
        """
        frequencies = self._get_frequencies_dict(chords)
        simplices_by_step = sorted(self._get_filtration(frequencies).items(),
                                   key=lambda x: x[0])

        chords_filtration = Filtration()

        step: Tuple
        for step in simplices_by_step:
            chords_filtration.setIndex(step[0])

            simplices = step[1]
            simplices.sort()
            simplices.sort(key=len)

            for simplex in simplices:
                s_id = "".join(simplex)
                if len(simplex) == 1:
                    chords_filtration.addSimplex(id=s_id)
                elif len(simplex) == 2:
                    chords_filtration.addSimplex(id=s_id, fs=simplex)
                else:
                    fs = set(["".join(sorted([simplex[i - 1], simplex[i]])) for i in range(len(simplex))])
                    chords_filtration.addSimplex(id=s_id, fs=fs)

        return chords_filtration

    # ---------- Tonnetz trajectory ----------

    def _dist(self, x: int, y: int) -> int:
        """
        distance between 2 pitch classes in the Tonnetz space

        :param x: :param y: pitch classes
        :return: 0 if this is the same pc, 1 if the interval between pc-s is represented in the tonnetz as is,
        2 otherwise
        """
        if x == y:
            return 0
        if (x - y) % 12 in self._tonnetz or (y - x) % 12 in self._tonnetz:
            return 1
        return 2

    def _neigh(self, x: int, y: int) -> bool:
        """
        defines whether two pc-s are neighbours with respect to distance defined above

        :param x, y: pitch classes
        :return: true if distance == 1
        """
        return self._dist(x, y) == 1

    def _neigh_to_lst(self, x: int, y_lst: List[int]) -> List[int]:
        """
        get all elements of y_lst which are neighbours with pitch class x

        :param x: pc
        :param y_lst: list of pc-s to check
        :return: list of neighbours
        """
        return [y for y in y_lst if self._neigh(x, y)]

    def _find_neigh(self, to_place: List[int], placed: VerticesDict) -> Union[Point, None]:
        """
        having positions of already placed pc-s and a list of pc-s to place, finds a pair of neighbours among them

        :param to_place: list of pitch classes
        :param placed: dict of already placed pitch classes with their coordinates in Tonnetz space
        (point {0,0} corresponds to pc = 0, or C in the current notation)
        :return: pair of neighbours, one of which is already placed and the other is not
        """
        for x in placed.keys():
            neighs = self._neigh_to_lst(x, to_place)
            if len(neighs) > 0:
                return x, neighs[0]

        return None

    def _place_neigh_pc(self, x: int, pos: Tuple[int, int], y: int) -> Tuple[int, int]:
        """
        depending on the interval between x and y defines y position according to x position

        :param x: :param y: pitch classes
        :param pos: coordinates of previously placed x
        :return: position of y
        :raises: ValueError if x and y are not neighbours in the chosen Tonnetz basis
        """
        x_pos, y_pos = pos
        int_m1 = (x - y) % 12
        int_p1 = (y - x) % 12
        if int_m1 == self._tonnetz_axis["x"]:
            return x_pos - 1, y_pos
        if int_p1 == self._tonnetz_axis["x"]:
            return x_pos + 1, y_pos
        if int_m1 == self._tonnetz_axis["y"]:
            return x_pos, y_pos - 1
        if int_p1 == self._tonnetz_axis["y"]:
            return x_pos, y_pos + 1
        if int_m1 == self._tonnetz_axis["xy"]:
            return x_pos + 1, y_pos + 1
        if int_p1 == self._tonnetz_axis["xy"]:
            return x_pos - 1, y_pos - 1

        raise ValueError("PC are not neighbours in the chosen tonnetz basis")

    def _place_not_neigh_pc(self, placed: VerticesDict, y: int) -> Point:
        """
        depending on the interval between x and y defines y position according to x position if x and y are not
        neighbours for Tonnetz {3,4,5} that can happen in two cases: - diminished triads - placing the first note of
        one chord with respect to another chord inside triads other than diminished only intervals equal to tonnetz
        intervals can appear the algorithm is to find the shared neighbour and place the note depending on it

        :param placed: positions of already placed pitch classes
        :param y: pc to place
        :return: position of y
        :raises: ValueError if it is impossible to find shared neighbour (i.e. something went competely wrong)
        """
        x_int = self._tonnetz_axis["x"]
        y_int = self._tonnetz_axis["y"]
        xy_int = self._tonnetz_axis["xy"]

        for x, pos in placed.items():
            x_pos, y_pos = pos

            if x == y:
                return x_pos, y_pos
            # the only one that really can happen inside chord
            if (x + 2 * y_int) % 12 == y:
                return x_pos, y_pos + 2
            if (x - 2 * y_int) % 12 == y:
                return x_pos, y_pos - 2
            # can happen when placing ref point from one chord in accordance with another
            if (x - x_int + xy_int) % 12 == y:
                return x_pos + 1, y_pos - 1
            if (x + x_int - xy_int) % 12 == y:
                return x_pos - 1, y_pos + 1

            if (x + y_int - xy_int) % 12 == y:
                return x_pos + 1, y_pos + 2
            if (x - y_int + xy_int) % 12 == y:
                return x_pos - 1, y_pos - 2
        raise ValueError("In T(3,4,5) other intervals that can be found in triads (maj, min, aug, dim) "
                         "should be expressed via neighbouring relations")

    def _place_pc(self, to_place: List[int], placed: VerticesDict) -> VerticesDict:
        """
        places list of pitch classes depending on already placed pc(s) with position(s)

        :param to_place: pc to place: list(int)
        :param placed: already placed pc(s) with position(s): dict(int: (int, int)) = dict(pc:(x_pos, y_pos))
        :return: dict of positions with both previously and newly placed pc-s
        """
        neighbours = self._find_neigh(to_place, placed)
        while neighbours is not None:
            x = neighbours[0]
            x_pos = placed[x]
            y = neighbours[1]

            y_pos = self._place_neigh_pc(x, x_pos, y)
            placed[y] = y_pos
            to_place.remove(y)

            neighbours = self._find_neigh(to_place, placed)

        while len(to_place) > 0:
            y = to_place[0]
            y_pos = self._place_not_neigh_pc(placed, y)
            placed[y] = y_pos
            to_place.remove(y)
        return placed

    def _get_reference_y_pitch_class(self, x_chord: Sequence[int], y_chord: Sequence[int]) -> int:
        """
        find the pc from y-chord closest to all the pc in x-chord

        :param x_chord: chord
        :param y_chord: chord to find reference pc in
        :return: reference pitch class (i.e. the one the chord x pc can be positioned on)
        """
        cur_min = 7
        cur_pc = None
        for y in y_chord:
            sum_xy_dist = sum([self._dist(x, y) for x in x_chord])
            if sum_xy_dist < cur_min:
                cur_min = sum_xy_dist
                cur_pc = y
        if cur_pc is None:
            raise ValueError("should choose at least 1 note")
        return cur_pc

    def _get_reference_pitch_classes(self, x_chord: ArrayLike, y_chord: ArrayLike) -> Tuple[int, int]:
        """
        finds a pair of pitch classes one from each chord such that dist(pc_x, chord_y) is the smallest

        :param x_chord: :param y_chord: chords represented as a collection of pitches like [0,4,7]
        :return: pair of pitch classes
        """
        return self._get_reference_y_pitch_class(y_chord, x_chord), self._get_reference_y_pitch_class(x_chord, y_chord)

    def _place_first_chord(self, chord: Sequence[int]) -> VerticesDict:
        """
        finds the position for the first chord in the initial coordinates (rectangle 3x4 with C = (0,0))

        :param chord: list of pc in chord
        :return: dict with positions of all pc in chord
        """
        x = chord[0]
        pos = self._initial_grid[x]
        to_place = set(chord) - {x}
        placed = {x: pos}
        return self._place_pc(list(to_place), placed)

    def _place_chord_on_prev(self, prev_placed: VerticesDict, to_place: List[int]) -> VerticesDict:
        """
        finds the position for the pitch classes in chord depending on previously placed pitch classes

        :param prev_placed: dict of previously placed pc mapped to their positions
        :param to_place: list of pc to place
        :return: position of chord to_place
        """
        x_ref, y_ref = self._get_reference_pitch_classes(list(prev_placed.keys()), to_place)
        y_pos = self._place_pc([y_ref], {x_ref: prev_placed[x_ref]})

        if len(y_pos.keys()) > 1:
            y_pos.pop(x_ref)
        return self._place_pc(list(set(to_place) - {y_ref}), y_pos)

    def _construct_tonnetz_for_chords(self, chords: List[VerticesDict]) -> nx.Graph:
        """
        construct graph representation of Tonnetz for given chords

        :param chords: list of chords with already calculated positions
        :return: Tonnetz graph
        """
        tonnetz = nx.Graph()
        for chord_positions in chords:
            intervals = list(
                filter(lambda x: (x[0] - x[1]) % 12 in self._tonnetz or (x[1] - x[0]) % 12 in self._tonnetz,
                       combinations(chord_positions.keys(), 2)))
            for interval in intervals:
                points = [chord_positions[i] for i in interval]

                for note in points:
                    tonnetz.add_node(note)
                tonnetz.add_edge(points[0], points[1])
        return tonnetz

    @staticmethod
    def _get_compact_tonnetz(t1: nx.Graph, t2: nx.Graph) -> int:
        """
        compares to Tonnetz graphs in terms of compactness (number of connected components and diameter)

        :param t1: :param t2: Tonnetz graph
        :return: 0 if the first graph is more compact 1 otherwise
        """
        cc1 = len(list(nx.connected_components(t1)))
        cc2 = len(list(nx.connected_components(t2)))
        if cc1 < cc2:
            return 0
        if cc1 > cc2:
            return 1

        if nx.is_connected(t1) and nx.is_connected(t2):
            d1 = nx.diameter(t1)
            d2 = nx.diameter(t2)

            if d1 < d2:
                return 0
            return 1

        return 0

    def _place_chord_optimal(self, prev_placed0: VerticesDict, prev_placed1: VerticesDict, to_place: List[int],
                             next_to_place: List[int]) -> VerticesDict:
        """
        finds the optimal position for the chord with respect to previously_previously placed chord and the next chord
        (i.e. to place chord C_n it considers C_{n-2}, C_{n-1} and C_{n+1})
        The algorithm:
        - find position of C_n depending on C_{n-1}
        - find position of C_{n+1} depending on C_{n-1}
        - find position of C_n depending on C_{n+1} (as they are neighbours in the chords sequence)
        - compare compactness of the grapsh {C_{n-2}, C_{n-1}, C_n} and {C_{n-2}, C_{n-1}, C_n'}

        :param prev_placed0: chord C_{n-2}
        :param prev_placed1: chord C_{n-1}
        :param to_place: chord C_n
        :param next_to_place: chord C_{n+1}
        :return: optimal position of C_n
        """
        pos0 = self._place_chord_on_prev(prev_placed1, to_place)
        pos_next = self._place_chord_on_prev(prev_placed1, next_to_place)
        pos1 = self._place_chord_on_prev(pos_next, to_place)

        tonnetz = self._construct_tonnetz_for_chords([prev_placed0, prev_placed1, pos0])
        tonnetz1 = self._construct_tonnetz_for_chords([prev_placed0, prev_placed1, pos1])

        if self._get_compact_tonnetz(tonnetz, tonnetz1) == 0:
            return pos0
        return pos1

    def _compute_trajectory(self, chords: List[ArrayLike]) -> List[VerticesDict]:
        """
        get positions for all the chords

        :param chords: sequence of chords (each is a list of pc)
        :return: the trajectory, i.e. the list of chords for each of which the positions of pc are given
        """
        trajectory = [self._place_first_chord(chords[0])]

        if len(chords) == 1:
            self._trajectory = trajectory
            return self._trajectory

        trajectory.append(self._place_chord_on_prev(trajectory[0], chords[1]))

        if len(chords) == 2:
            self._trajectory = trajectory
            return self._trajectory

        for i in range(2, len(chords) - 1):
            trajectory.append(self._place_chord_optimal(trajectory[i - 2], trajectory[i - 1], chords[i], chords[i + 1]))

        trajectory.append(self._place_chord_on_prev(trajectory[len(chords) - 2], chords[-1]))

        self._trajectory = trajectory
        return self._trajectory

    def plt_trajectory(self) -> NoReturn:
        """
        plot the trajectory using matplotlib
        """
        assert self._trajectory is not None

        for positions in self._trajectory:

            intervals = list(
                filter(lambda i: (i[0] - i[1]) % 12 in self._tonnetz or (i[1] - i[0]) % 12 in self._tonnetz,
                       combinations(positions.keys(), 2)))

            for interval in intervals:
                points = [positions[i] for i in interval]

                x = [point[0] for point in points]
                y = [point[1] for point in points]

                plt.plot(x, y, "r-")

        plt.grid()
        ax = plt.gca()
        ax.set(xlabel='major third', ylabel='minor third')
        ax.set_title("Tonnetz trajectory")

        ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        plt.show()

    @staticmethod
    def _get_key(positions: List[Point]) -> str:
        """
        get id for the simplex with given vertices / edges
        :param positions: coordinates of vertices
        :return: string id representation
        """
        str_values = []
        for pos in positions:
            str_values.append(str(pos[0]) + "_" + str(pos[1]))

        return ";".join(str_values)

    def _get_simplices_from_trajectory(self, durations: ArrayLike) \
            -> dict[Hashable, float]:
        """
        get dict of simplices with their weights from the trajectory

        :param durations: list of durations for each chord
        :return: dict of simplices with their normalized weight
        """
        simplices_0d = {}  # pc
        simplices_1d = {}  # intervals
        simplices_2d = {}  # chords

        for i in range(len(self._trajectory)):
            positions = self._trajectory[i]
            duration = durations[i]

            for pos in positions.values():
                key = self._get_key([pos])
                simplices_0d[key] = simplices_0d.get(key, 0) + duration

            intervals = list(
                filter(lambda x: (x[0] - x[1]) % 12 in self._tonnetz or (x[1] - x[0]) % 12 in self._tonnetz,
                       combinations(positions.keys(), 2)))
            intervals = [sorted(i) for i in intervals]

            for interval in intervals:
                key = self._get_key([positions[interval[0]], positions[interval[1]]])
                simplices_1d[key] = simplices_1d.get(key, 0) + duration

            # if the chord is fully connected in the given Tonnetz (i.e. major or minor third)
            if len(intervals) == 3:
                interval_sorted = sorted(list(positions.keys()))
                key = self._get_key(
                    [positions[interval_sorted[0]], positions[interval_sorted[1]], positions[interval_sorted[2]]])

                simplices_2d[key] = simplices_2d.get(key, 0) + duration
        simplices = dict(simplices_0d, **simplices_1d)
        simplices.update(simplices_2d)

        return normalize_dict_values(simplices)

    def get_filtration_for_trajectory(self, dict_durations: Dict[Hashable, float]) -> Filtration:
        """
        build Filtration on the simplices dict

        :param dict_durations: dict of simplices with their weights
        :return: Filtration
        """
        simplices_by_step = sorted(self._get_filtration(dict_durations).items(), key=lambda x: x[0])

        filtration = Filtration()

        step: Tuple
        for step in simplices_by_step:
            filtration.setIndex(step[0])

            simplices = step[1]

            simplices_edges = [s.split(";") for s in simplices]
            simplices_edges.sort(key=len)  # sort by length

            for i in range(len(simplices)):
                s_id = simplices[i]
                simplex = simplices_edges[i]
                if len(simplex) == 1:
                    filtration.addSimplex(id=s_id)
                elif len(simplex) == 2:
                    filtration.addSimplex(id=s_id, fs=simplex)
                else:
                    # dict of durations contains only the elements that WILL be in SC, but when calculating intervals
                    # for chord we cannot assert the order of pc in interval. It is not significant in general,
                    # but should be kept in mind since we need to determine the same interval id as the one already
                    # in filtration
                    fs = set(
                        filter(lambda x: x in dict_durations, [";".join(x) for x in list(permutations(simplex, 2))]))
                    filtration.addSimplex(id=s_id, fs=fs)

        return filtration

    def _build_filtration_on_tonnetz_trajectory(self, chords: ArrayLike,
                                                durations: ArrayLike) -> Filtration:
        """
        build filtration for sequence of chords with durations

        :param chords: sequence of chords
        :param durations: chords durations
        :return: Filtration
        """
        assert len(chords) == len(durations)

        chords_unmapped = [self._get_pc_indexes(chord) for chord in chords]

        self._compute_trajectory(chords_unmapped)
        simplices = self._get_simplices_from_trajectory(durations)

        return self.get_filtration_for_trajectory(simplices)

    # ---------- Persistence ----------

    def _get_simplices_of_order(self, order: int) -> List:
        """
        find all simplices of given order with respect to their birth time

        :param order: order (or dimension) of simplex
        :return: list of simplexes of given order
        """
        assert self._filtration is not None

        simplices = []
        simplices_of_order_set = self._filtration.simplicesOfOrder(order)
        for simplex in self._filtration.simplices():
            if simplex in simplices_of_order_set:
                simplices.append(simplex)
        return simplices

    def compute_persistence(self, k=1) -> ArrayLike:
        """
        compute persistence diagram on the filtration

        :param k: over which dimension to compute diagram
        :return: array of birth-death pairs
        """
        assert self._filtration is not None

        gaussian_form = get_gauss_form(self._filtration.boundaryOperator(k=k))
        low_indexes = {i: get_low(gaussian_form[:, i]) for i in range(gaussian_form.shape[1])}

        simplices_k = self._get_simplices_of_order(k)
        simplices_km1 = self._get_simplices_of_order(k - 1)

        pairs = [(simplices_k[key], simplices_km1[val]) for key, val in low_indexes.items() if val != -1]

        self._persistence_diagram = np.array(list(filter(lambda x: x[0] != x[1],
                                    [[self._filtration.addedAtIndex(pair[1]), self._filtration.addedAtIndex(pair[0])]
                                     for pair in pairs])))
        return self._persistence_diagram

    def compute_persistence_merged(self, k_lim = 4) -> ArrayLike:
        persistence_diagrams = []
        for k in range(1, k_lim + 1):
            pers = self.compute_persistence(k)
            if len(pers) != 0:
                persistence_diagrams.append(self.compute_persistence(k))

        if len(persistence_diagrams) == 1:
            self._persistence_diagram = persistence_diagrams[0]
        elif len(persistence_diagrams) > 0:
            self._persistence_diagram = np.concatenate(persistence_diagrams)

        return self._persistence_diagram

    def plt_persistence(self) -> NoReturn:
        assert self._persistence_diagram is not None

        x = [interval[0] for interval in self._persistence_diagram]
        y = [interval[1] for interval in self._persistence_diagram]

        max_x = max(x)
        max_y = max(y)

        if max_x > max_y:
            max_el = max_x
        else:
            max_el = max_y

        diagonal = [i for i in range(max_el)]

        sns.scatterplot(x=x, y=y, marker='o')
        sns.lineplot(x=diagonal, y=diagonal)

        plt.grid()
        ax = plt.gca()
        ax.set(xlabel='birth', ylabel='death')
        ax.set_title("Persistence diagram")

        ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        plt.show()

    def compute_persistence_image(self):
        assert self._persistence_diagram is not None

        self._pers_img = self._persimg.fit_transform(self._persistence_diagram, skew=True)
        return self._pers_img

    def plot_persistence_image(self):
        assert self._persistence_diagram is not None

        if self._pers_img is None:
            self._pers_img = self.compute_persistence_image()

        self._persimg.plot_image(self._pers_img)
