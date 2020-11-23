from collections import OrderedDict
from typing import TypeVar, Generic

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class KeyCollection(Generic[_KT, _VT]):
    def __init__(self):
        self._dictionary = OrderedDict()

    def __getitem__(self, key: _KT) -> _VT:
        if isinstance(key, int):
            return list(self._dictionary.values())[key]
        return self._dictionary[key]

    def _add(self, value: _VT):
        self._dictionary[value.id] = value

    def __contains__(self, key: _KT) -> bool:
        return self._dictionary.__contains__(key)

    def __len__(self) -> int:
        return self._dictionary.values().__len__()

    def __iter__(self):
        return self._dictionary.values().__iter__()

    def __next__(self):
        return self._dictionary.values().__next__()

    def _ipython_key_completions_(self):
        return list(self._dictionary.keys())
