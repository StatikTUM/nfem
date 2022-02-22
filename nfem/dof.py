from __future__ import annotations

from typing import Tuple


class Dof:

    def __init__(self, id: Tuple[str, str], value: float, is_active: bool = True, external_force: float = 0.0):
        self.id = id
        self.ref_value = value
        self.value = value
        self.is_active = is_active
        self.external_force = 0.0

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.id == other.id
        return self.id == other

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def delta(self) -> float:
        return self.value - self.ref_value

    @delta.setter
    def delta(self, value: float) -> None:
        self.value = self.ref_value + value
