from rdkit import Chem
from scikit_mol.descriptors import MolecularDescriptorTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler
from typing import Iterable, List
import pickle


class PropLL:
    def __init__(
        self, desc_list: List[str] = ["MolWt"], bandwidth: float = 0.1
    ) -> None:
        self._desc_list = desc_list
        self._bandwidth = bandwidth
        self._make_pipeline()
        self._properties = ["_desc_list", "_bandwidth", "_pipeline"]

    @property
    def desc_list(self):
        return self._desc_list

    @desc_list.setter
    def desc_list(self, desc_list: List[str]):
        self._desc_list = desc_list
        self._make_pipeline()  # Remake pipeline if desc_list is changed

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth: float):
        self._bandwidth = bandwidth
        self._make_pipeline()  # Remake pipeline if bandwidth is changed

    def _make_pipeline(self):
        featurizer = MolecularDescriptorTransformer(desc_list=self.desc_list)
        scaler = RobustScaler()
        kde = KernelDensity(bandwidth=self.bandwidth)
        self._pipeline = make_pipeline(featurizer, scaler, kde)

    def analyze_dataset(self, mols: Iterable[Chem.Mol]) -> None:
        "Gather statistics from the list of molecules"
        self._pipeline.fit(mols)

    def save(self, filename: str) -> None:
        save_dict = {"Model": self.__class__.__name__}
        save_dict.update(
            {property: getattr(self, property) for property in self._properties}
        )
        with open(filename, "wb") as f:
            pickle.dump(
                save_dict, f
            )  # Can't really serialize the scikit-mol pipeline as text, versioning problems can appear

    def load(self, filename: str) -> None:
        with open(filename, "rb") as f:
            savedict = pickle.load(f)
        assert (
            savedict["Model"] == self.__class__.__name__
        ), f"Save data is of model type {savedict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        [setattr(self, property, savedict[property]) for property in self._properties]

    def calculate_ll(self, mol: Chem.Mol) -> float:
        "Method that calculates LL for a single molecule"
        return self._pipeline.score_samples([mol])[0]

    def calculate_lls(self, mols: List[Chem.Mol]) -> List[float]:
        "Method that calculates LL for a single molecule"
        kdes = self._pipeline.score_samples(mols)
        return list(kdes.flatten())
