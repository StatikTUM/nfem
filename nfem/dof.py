class Dof:
    def __init__(self, value, is_active=True):
        self.reference_value = value
        self.value = value
        self.is_active = is_active

    def __iadd__(self, value):
        self.value += value
        return self

    def __isub__(self, value):
        self.value -= value
        return self

    @property
    def delta(self):
        return self.value - self.reference_value

    @delta.setter
    def delta(self, value):
        self.value = self.reference_value + value
