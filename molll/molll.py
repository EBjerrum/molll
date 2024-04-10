from abc import ABC, abstractmethod
from typing import Iterable, List
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter, defaultdict
import math
import numpy as np
import json


class BaseLL(ABC):
    def __init__(
        self,
        smoothing: float = 0.1,
        estimated_keyspace: int = 2e6,
        alpha: float = 1,
        radius: int = 1,
    ) -> None:
        # Public properties affect LL calculation, but not the statistics collection
        self.smoothing = smoothing  # laplace smoothing prior
        self.estimated_keyspace = estimated_keyspace  # for laplace smoothing
        self.alpha = alpha  # Atom count correction in range 0 to 1

        # Private properties affect the key_data and should thus only be changed, if a refit is run
        self._radius = radius
        self._key_data = None  # The data structure for the observation data
        self._n_observations = 0

        self._properties = [
            "smoothing",
            "estimated_keyspace",
            "alpha",
            "_radius",
            "_n_observations",
        ]

    def _get_fp(self, mol: Chem.Mol) -> dict:
        fp = AllChem.GetMorganFingerprint(mol, self._radius)
        return fp.GetNonzeroElements()

    def save(self, filename: str) -> None:
        save_dict = self.get_savedict()  # Make abstract funcition
        json.dump(save_dict, open(filename, "w"), indent=2)

    def get_savedict(self):
        save_dict = {"Model": self.__class__.__name__}
        save_dict.update(
            {property: getattr(self, property) for property in self._properties}
        )
        save_dict["_key_data"] = self._get_keydata_as_dict()

        return save_dict
        # with open(filename, 'wb') as f:
        #     pickle.dump(self.key_data, f)

    def load(self, filename: str) -> None:
        save_dict = json.load(open(filename, "r"))
        self.set_savedict(save_dict)

    def set_savedict(self, savedict):
        assert (
            savedict["Model"] == self.__class__.__name__
        ), f"Save data is of model type {savedict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        [setattr(self, property, savedict[property]) for property in self._properties]
        self._set_keydata_from_dict(savedict["_key_data"])

    @abstractmethod
    def _get_keydata_as_dict(self) -> dict:
        "convert keydata object to serializable dictionary"

    @abstractmethod
    def _set_keydata_from_dict(self, keydata: dict) -> None:
        "convert serialized keydata dict to correct python object"

    @abstractmethod
    def calculate_ll(self, mol: Chem.Mol) -> float:
        "Method that calculates LL for"

    @abstractmethod
    def fit(self, mols: Iterable[Chem.Mol]) -> None:
        "Gather statistics from the list of molecules"

    def calculate_lls(self, mols: List[Chem.Mol]) -> List[float]:
        return [self.calculate_ll(mol) for mol in mols]


# %%
class AtomLL(BaseLL):
    def fit(self, mols: Iterable[Chem.Mol]) -> None:
        self._key_data = Counter()
        for mol in mols:
            self._key_data.update(self._get_fp(mol))
        self._calculate_observations()

    def _calculate_observations(self) -> None:
        self._n_observations = sum(self._key_data.values())

    @property
    def observed_keyspace(self):
        return len(self._key_data)

    def _get_keydata_as_dict(self) -> dict:
        return dict(self._key_data)

    def _set_keydata_from_dict(self, keydict: dict) -> None:
        keydict = {int(key): value for key, value in keydict.items()}
        self._key_data = Counter(keydict)

    def calculate_ll(self, mol: Chem.Mol) -> None:
        ll = 0.0
        fp = self._get_fp(mol)
        atom_count = 0

        for key, num_atoms in fp.items():
            stat_count = self._key_data[key]  # Returns 0 if not found
            # TODO, this can potentially precomputed in the fit for faster inference
            likelihood = (stat_count + self.smoothing) / (
                self._n_observations + self.estimated_keyspace * self.smoothing
            )
            ll += math.log(likelihood) * num_atoms
            atom_count += num_atoms

        # Correction for number of atoms
        # divide by atom_count^alpha, alpha between 0 (no correction) to 1 (full correction)
        ll_corrected = ll / (atom_count**self.alpha)

        return ll_corrected


class MolLL(BaseLL):
    def fit(self, mols: Iterable[Chem.Mol]) -> None:
        self._key_data = defaultdict(
            lambda: Counter()
        )  # We always get a counter, but if no key and empty counter we get a zero back.
        for mol in mols:
            fp = self._get_fp(mol)
            for key, count in fp.items():
                self._key_data[key].update([count])  # We are counting the counts, lol
            # Add sample_counts
            self._key_data["SampleObservationSum"].update(
                [sum(fp.values())]
            )  # Make it possible to see how many keys are normal to see in a mol

        self._calculate_observations()
        self._smooth_all_counters()

    def _calculate_observations(self) -> None:
        n_observations = 0
        for counter in self._key_data.values():
            for count, specific_obs in counter.items():
                n_observations += count * specific_obs
        self._n_observations = n_observations

    @property
    def observed_keyspace(self):
        return sum([len(counter) for counter in self._key_data.values()])

    def _get_keydata_as_dict(self) -> dict:
        "convert keydata object to serializable dictionary"
        return {key: dict(counter) for key, counter in self._key_data.items()}

    def _set_keydata_from_dict(self, keydata: dict) -> None:
        "convert serialized keydata dict to correct python object"
        key_data_object = defaultdict(lambda: Counter())
        # json only supports str keys, need to back_convert
        for fpkey, counterdict in keydata.items():
            if fpkey != "SampleObservationSum":
                fpkey = int(fpkey)
            counterdict = {int(key): counts for key, counts in counterdict.items()}
            key_data_object[fpkey] = Counter(counterdict)
        self._key_data = key_data_object

    def _smooth_counter(self, counter: Counter) -> Counter:
        smoothed_counter = Counter()

        smoothed_counter[1] = counter[1]

        for key in range(2, max(counter.keys()) + 1):
            neighbor_values = (
                np.array([counter[key - 1], counter[key], counter[key + 1]]) + 0.15
            )

            # Calculate the geometric mean of the neighbor values
            geometric_mean = math.ceil(np.exp(np.mean(np.log(neighbor_values))))

            # Append the smoothed value to the output
            smoothed_counter[key] = geometric_mean

        return smoothed_counter

    def _smooth_all_counters(self) -> None:
        for key, counter in self._key_data.items():
            self._key_data[key] = self._smooth_counter(counter)

    def calculate_ll(self, mol: Chem.Mol) -> None:
        ll = 0.0
        fp = self._get_fp(mol)
        atom_count = 0

        for key, counts in list(fp.items()) + [
            ("SampleObservationSum", sum(fp.values()))
        ]:
            count = self._key_data[key][counts]  # Returns 0 if not found
            num_atoms = counts
            # TODO, this can potentially precomputed in the fit for faster inference
            likelihood = (count + self.smoothing) / (
                self._n_observations + self.estimated_keyspace * self.smoothing
            )
            ll += math.log(likelihood) * num_atoms
            atom_count += num_atoms

        # Correction for number of atoms
        # divide by atom_count^alpha, alpha between 0 (no correction) to 1 (full correction)
        ll_corrected = ll / (atom_count**self.alpha)

        return ll_corrected
