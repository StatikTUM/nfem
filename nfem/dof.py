class Dof:
    def __init__(self, value, is_active=True, external_force=0.0):
        self.reference_value = value
        self.value = value
        self.is_active = is_active
        self.external_force = 0.0

    @property
    def delta(self):
        return self.value - self.reference_value

    @delta.setter
    def delta(self, value):
        self.value = self.reference_value + value
