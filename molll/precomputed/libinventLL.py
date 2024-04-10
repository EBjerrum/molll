import os
import json
from molll import AtomLL, MolLL


thisfiledir = os.path.dirname(os.path.abspath(__file__))

class LibInventAtomLLr1(AtomLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/AtomLL_libinvent_radius_1.json", 'r'))
        assert save_dict["Model"] == "AtomLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)

class LibInventAtomLLr2(AtomLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/AtomLL_libinvent_radius_2.json", 'r'))
        assert save_dict["Model"] == "AtomLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)


class LibInventAtomLLr3(AtomLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/AtomLL_libinvent_radius_3.json", 'r'))
        assert save_dict["Model"] == "AtomLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)


class LibInventMolLLr1(MolLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/MolLL_libinvent_radius_1.json", 'r'))
        assert save_dict["Model"] == "MolLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)


class LibInventMolLLr2(MolLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/MolLL_libinvent_radius_2.json", 'r'))
        assert save_dict["Model"] == "MolLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)


class LibInventMolLLr3(MolLL):
    def __init__(self):
        super().__init__()
        save_dict = json.load(open(f"{thisfiledir}/data/MolLL_libinvent_radius_3.json", 'r'))
        assert save_dict["Model"] == "MolLL", f"Save data is of model type {save_dict['Model']}, which can't be loaded into a model of class {self.__class__.__name__}"
        save_dict["Model"] = self.__class__.__name__
        self.set_savedict(save_dict)
