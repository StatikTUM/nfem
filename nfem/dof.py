"""Degree of freedom of a mechanical model."""

from __future__ import annotations

from typing import Tuple


class Dof:
    """Degree of freedom of a mechanical model."""

    def __init__(self, id: Tuple[str, str], value: float):
        """Create a new degree of freedom.

        :id: Unique ID.
        :value: Actual value.
        """
        self.id: Tuple[str, str] = id
        self.ref_value: float = value
        self.value: float = value
        self.is_active: bool = True
        self.external_force: float = 0.0
        self.residual: float = 0.0

    def __eq__(self, other) -> bool:
        """Compare for equality.

        :other: Object for comparison.
        :return: True when both objects are equal.
        """
        if isinstance(other, self.__class__):
            return self.id == other.id
        return self.id == other

    def __ne__(self, other) -> bool:
        """Compare for inequality.

        :other: Object for comparison.
        :return: True when both objects are not equal.
        """
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Hash code for comparisons."""
        return hash(self.id)

    @property
    def delta(self) -> float:
        """Get or set the difference between actual and reference value."""
        return self.value - self.ref_value

    @delta.setter
    def delta(self, value: float) -> None:
        self.value = self.ref_value + value
