class Dof:
    def __init__(self, key, value, is_active=True, external_force=0.0):
        self.key = key
        self.reference_value = value
        self.value = value
        self.is_active = is_active
        self.external_force = 0.0

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key == other.key
        return self.key == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.key)

    @property
    def delta(self):
        return self.value - self.reference_value

    @delta.setter
    def delta(self, value):
        self.value = self.reference_value + value
