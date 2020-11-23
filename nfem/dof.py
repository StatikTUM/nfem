class Dof:
    def __init__(self, id, value, is_active=True, external_force=0.0):
        self.id = id
        self.ref_value = value
        self.value = value
        self.is_active = is_active
        self.external_force = 0.0

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return self.id == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    @property
    def delta(self):
        return self.value - self.ref_value

    @delta.setter
    def delta(self, value):
        self.value = self.ref_value + value
