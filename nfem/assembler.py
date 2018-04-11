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

    def Calculate(self, u, lam, system_k_e, system_k_u, system_k_g, system_f):
        """FIXME"""

        for element, indices in self.element_freedom_table:
            element_u = np.array([u[index] for index in indices])

            element_k_e, element_k_u, element_k_g, element_f = element.Calculate(element_u, lam)

            for element_row, system_row in enumerate(indices):
                for element_col, system_col in enumerate(indices):
                    if element_k_e is not None:
                        system_k_e[system_row, system_col] += element_k_e[element_row, element_col]
                    if element_k_u is not None:
                        system_k_u[system_row, system_col] += element_k_u[element_row, element_col]
                    if element_k_g is not None:
                        system_k_g[system_row, system_col] += element_k_g[element_row, element_col]

                if element_f is not None:
                    system_f[system_row] += element_f[element_row]
