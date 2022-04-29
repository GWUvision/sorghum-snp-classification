from types import SimpleNamespace
from collections.abc import Mapping
class HParam(SimpleNamespace, Mapping):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
 