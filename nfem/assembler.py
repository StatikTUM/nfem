"""FIXME"""

import numpy as np

class Assembler(object):
    """FIXME"""

    def __init__(self, model):
        """FIXME"""

        # --- dof indices

        processed_dofs = set()

        free_dofs = list()
        fixed_dofs = list()

        for element in model.elements.values():
            for dof in element.Dofs():
                if dof in processed_dofs:
                    continue
                else:
                    processed_dofs.add(dof)

                if dof in model.dirichlet_conditions:
                    fixed_dofs.append(dof)
                else:
                    free_dofs.append(dof)

        dofs = free_dofs + fixed_dofs

        dof_indices = {dof: index for index, dof in enumerate(dofs)}


        # --- element freedom table

        element_freedom_table = list()

        for element in model.elements.values():
            indices = [dof_indices[dof] for dof in element.Dofs()]

            element_freedom_table.append((element, indices))


        # --- store

        self.dofs = dofs
        self.dof_indices = dof_indices
        self.dof_count = len(dofs)
        self.free_dof_count = len(free_dofs)
        self.fixed_dof_count = len(fixed_dofs)
        self.element_freedom_table = element_freedom_table

    def IndexOfDof(self, dof):
        """FIXME"""

        return self.dof_indices[dof]

    def DofAtIndex(self, index):
        """FIXME"""

        return self.dofs[index]

    def AssembleMatrix(self, system_matrix, calculate_element_matrix):
        """FIXME"""

        for element, indices in self.element_freedom_table:
            element_matrix = calculate_element_matrix(element)

            if element_matrix is None:
                continue

            for element_row, system_row in enumerate(indices):
                for element_col, system_col in enumerate(indices):
                    system_matrix[system_row, system_col] += element_matrix[element_row, element_col]

    def AssembleVector(self, system_vector, calculate_element_vector):
        """FIXME"""

        for element, indices in self.element_freedom_table:
            element_vector = calculate_element_vector(element)

            if element_vector is None:
                continue

            for element_row, system_row in enumerate(indices):
                system_vector[system_row] += element_vector[element_row]
