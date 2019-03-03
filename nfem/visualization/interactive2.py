"""
Interactive
===========

Module to initiate the NFEM-Teching-Tool GUI
"""

import traceback

from .python_ui import ApplicationWindow, Option, Fore, Style
from .GUI_classes import SideBySide2D3DPlots, AnalysisTab, VisualisationTab, AnimationWindow
from ..bracketing import bracketing
from .plot import (plot_model, plot_bounding_cube, plot_history_curve,
                   plot_crosshair, plot_scaled_model, get_bounding_box)

def interact2(model, dof):
    """
    Launch the interactive GUI

    Parameters
    ----------
    model : object
        An object of class Model
    dof : tuple
        a tuple of the node ID and dof type

    Returns
    -------
    model : object
        the model after the GUI is closed

    Examples
    --------
        >>> model = interact(model=model, dof=(42, 'u'))
    """
    window = MainWindow.run(model=model, dof=dof)
    return window.model


class MainWindow(ApplicationWindow):
    """
    Main Application window.

    Parameters
    ----------
    model : Model
        An object of the class model
    dof : tuple
        A tuple of the node id and the dof type
    """

    def __init__(self, model, dof):
        super(MainWindow, self).__init__(
            title=f'NFEM Teaching Tool (Model: {model.name})',
            content=SideBySide2D3DPlots(self._draw))

        self.branches = [model]
        self.dof = dof

        # Check if the given dof is a free dof in the model
        try:
            dof_index = model.free_dofs.index(dof)
        except ValueError:
            self.WARNING(f'Selected dof {dof} is not part of the free_dofs: {model.free_dofs}')
            self.WARNING(f'Running the tool with the first free dof: {model.free_dofs[0]}')
            dof_index = 0

        self.options = dict()
        # == analysis options
        self.options['tab_idx'] = Option(0)
        self.options['solver_idx'] = Option(0)

        self.options['linear/lambda'] = Option(0)

        self.options['nonlinear/predictor_idx'] = Option(0)
        self.options['nonlinear/predictor/tangential_flag'] = Option(True)
        self.options['nonlinear/predictor/lambda'] = Option(0.0)
        self.options['nonlinear/predictor/delta_lambda'] = Option(0.0)
        self.options['nonlinear/predictor/dof_idx'] = Option(dof_index)
        self.options['nonlinear/predictor/dof_value'] = Option(0.0)
        self.options['nonlinear/predictor/delta-dof'] = Option(0.0)
        self.options['nonlinear/predictor/increment_length'] = Option(0.0)
        self.options['nonlinear/predictor/beta'] = Option(1.0)

        self.options['nonlinear/constraint_idx'] = Option(0)
        self.options['nonlinear/constraint/dof_idx'] = Option(dof_index)

        self.options['nonlinear/newtonraphson/maxiterations'] = Option(100)
        self.options['nonlinear/newtonraphson/tolerance_power'] = Option(-7)

        self.options['nonlinear/solution/det(K)_flag'] = Option(False)
        self.options['nonlinear/solution/eigenproblem_flag'] = Option(False)

        self.options['bracketing/maxiterations'] = Option(100)
        self.options['bracketing/tolerance_power'] = Option(-7)

        self.options['plot/highlight_dof'] = Option(True, self.redraw)
        self.options['plot/dirichlet'] = Option(True, self.redraw)
        self.options['plot/neumann'] = Option(True, self.redraw)
        self.options['plot/symbol_size'] = Option(5, self.redraw)
        self.options['plot/eigenvector_flag'] = Option(False, self.redraw)
        self.options['plot/dof_idx'] = Option(dof_index, self.redraw)
        self.options['plot/load_disp_curve_flag'] = Option(True, self.redraw)
        self.options['plot/load_disp_curve_iter_flag'] = Option(False, self.redraw)
        self.options['plot/det(K)_flag'] = Option(False, self.redraw)
        self.options['plot/eigenvalue_flag'] = Option(False, self.redraw)

    @property
    def option_values(self):
        """ dict of self.options values instead of objects """
        return {option[0] : option[1].value for option in self.options.items()}

    @property
    def solvers(self):
        return ['nonlinear', 'LPB', 'linear', 'bracketing']

    @property
    def predictors(self):
        return [
            'lambda',
            'delta-lambda',
            'dof',
            'delta-dof',
            'arc-length',
            'arclength_eigenvector'
        ]

    @property
    def constraints(self):
        return ['load-control', 'displacement-control', 'arc-length-control']

    @property
    def model(self):
        return self.branches[-1]

    @model.setter
    def model(self, value):
        self.branches[-1] = value

    def DEBUG(self, message=None):
        """ print to console """
        if message:
            print(message)

    def DEBUG_red(self, message=None):
        """ print to console in red color"""
        if message:
            print(Fore.RED + message + Style.RESET_ALL)

    def DEBUG_blue(self, message=None):
        """ print to console in blue color"""
        if message:
            print(Fore.CYAN + message + Style.RESET_ALL)

    def WARNING(self, message=None):
        """ print WARNING to console"""
        if message:
            print(Fore.YELLOW + "WARNING: " + message + Style.RESET_ALL)

    def _build_sidebar(self, builder):
        builder.add_tabs(
            items=[
                ('Analysis', AnalysisTab),
                ('Visualisation', VisualisationTab)],
            option=self.options['tab_idx'])
        builder.add_stretch()
        builder.add_button(
            label='Solve',
            action=self.solve_click)
        builder.add_button(
            label='Go back',
            action=self.go_back_click)
        builder.add_button(
            label='Reset path',
            action=self.reset_path_click)
        builder.add_button(
            label='New path',
            action=self.new_path_click)
        builder.add_button(
            label='Reset all',
            action=self.reset_all_click)

    def _draw(self, ax3d, ax2d):
        # get variables
        options = self.option_values
        dof = self.model.free_dofs[options['plot/dof_idx']]

        # plot bounding cube
        bounding_box = get_bounding_box(self.model.get_model_history())
        plot_bounding_cube(ax3d, bounding_box)

        # plot initial and deformed models
        plot_model(ax3d, self.model, 'gray', True, **options)
        plot_model(ax3d, self.model, 'red', False, **options)

        # plot eigenvector
        if options['plot/eigenvector_flag'] and self.model.first_eigenvector_model is not None:
            plot_scaled_model(ax3d, self.model.first_eigenvector_model, 'green')

        # logger
        logger = LoadDisplacementLogger(dof)
        label = logger.xlabel + " : " + logger.ylabel
        ax2d.set(xlabel=logger.xlabel, ylabel=logger.ylabel, title=logger.title)
        ax3d.set(
            xlabel='< x >',
            ylabel='< y >',
            zlabel='< z >',
        )

        # plot load displacement curve
        if options['plot/load_disp_curve_flag']:
            # other branches at first level
            n_branches = len(self.branches)
            for i, branch_model in enumerate(self.branches[:-1]):
                grey_level = i/float(n_branches)
                plot_history_curve(
                    ax=ax2d,
                    model=branch_model,
                    xy_function=logger,
                    fmt='--x',
                    label=f'Branch {i+1} of {n_branches}',
                    color=str(grey_level))
            # main branch
            plot_history_curve(
                ax=ax2d,
                model=self.model,
                xy_function=logger,
                fmt='-o',
                label=label,
                color='tab:blue')
            plot_crosshair(
                ax=ax2d,
                x=self.model.get_dof_state(dof),
                y=self.model.lam,
                linestyle='-.',
                color='tab:blue',
                linewidth=0.75)

        # load displacement iteration plot
        if options['plot/load_disp_curve_iter_flag']:
            plot_history_curve(
                ax=ax2d,
                model=self.model,
                xy_function=logger,
                fmt='--o',
                label=f'{label} (iter)',
                skip_iterations=False,
                linewidth=0.75,
                markersize=2.0,
                color='tab:orange')

        # det_k plot
        if options['plot/det(K)_flag']:
            logger = CustomLogger(
                x_fct=lambda model: model.get_dof_state(dof=dof),
                y_fct=lambda model: model.det_k,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Det(K)')
            plot_history_curve(
                ax=ax2d,
                model=self.model,
                xy_function=logger,
                fmt='-o',
                label=logger.title,
                color='tab:green')

        # eigenvalue plot
        if options['plot/eigenvalue_flag']:
            logger = CustomLogger(
                x_fct=lambda model: model.get_dof_state(dof=dof),
                y_fct=lambda model: None if not model.first_eigenvalue else model.first_eigenvalue*model.lam,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Eigenvalue')
            plot_history_curve(
                ax=ax2d,
                model=self.model,
                xy_function=logger,
                fmt='-o',
                label=logger.title,
                color='tab:red')

    def show_animation_click(self, builder):
        builder.show_dialog(AnimationWindow, size=(800, 800))

    def solve_click(self):
        try:
            solver = self.solvers[self.options['solver_idx'].value]

            if solver == 'linear':
                model = self.solve_linear(
                    model=self.model.get_duplicate(),
                    lam=self.options['linear/lambda'].value)

            elif solver == 'LPB':
                model = self.solve_LPB(
                    model=self.model.get_duplicate(),
                    lam=self.options['linear/lambda'].value)

            elif solver == 'nonlinear':
                model = self.solve_nonlinear(
                    model=self.model.get_duplicate(),
                    options=self.option_values)

            elif solver == 'bracketing':
                model = self.solve_bracketing(
                    model=self.model,
                    options=self.option_values)

            else:
                raise Exception(f'Unknown solver {solver}')

        except Exception as e:
            traceback.print_exc()
            self.show_error_dialog(str(e))
            return

        self.model = model
        self.options['nonlinear/predictor/increment_length'].change(model.get_increment_norm())
        self.redraw()

    def solve_linear(self, model, lam):
        model.lam = lam
        model.perform_linear_solution_step()
        return model

    def solve_LPB(self, model, lam):
        if model.get_previous_model().get_previous_model() is not None:
            raise RuntimeError('LPB can only be done on the initial model')
        model.lam = lam
        model.perform_linear_solution_step()
        model.solve_linear_eigenvalues()
        return model

    def solve_nonlinear(self, model, options):
        predictor = self.predictors[options['nonlinear/predictor_idx']]
        tangential_flag = options['nonlinear/predictor/tangential_flag']

        if predictor == 'lambda':
            value = options['nonlinear/predictor/lambda']
            if tangential_flag:
                model.predict_tangential(strategy=predictor, value=value)
            else:
                model.lam = value

        elif predictor == 'delta-lambda':
            value = options['nonlinear/predictor/delta_lambda']
            if tangential_flag:
                model.predict_tangential(strategy=predictor, value=value)
            else:
                model.lam += value

        elif predictor == 'dof':
            dof = model.free_dofs[options['nonlinear/predictor/dof_idx']]
            dof_value = options['nonlinear/predictor/dof_value']
            if tangential_flag:
                model.predict_tangential(strategy=predictor, dof=dof, value=dof_value)
            else:
                model.set_dof_state(dof, dof_value)

        elif predictor == 'delta-dof':
            dof = model.free_dofs[options['nonlinear/predictor/dof_idx']]
            dof_value_increment = options['nonlinear/predictor/delta-dof']
            if tangential_flag:
                model.predict_tangential(strategy=predictor, dof=dof, value=dof_value_increment)
            else:
                model.increment_dof_state(dof, dof_value_increment)

        elif predictor == 'arc-length':
            arclength = options['nonlinear/predictor/increment_length']
            model.predict_tangential(strategy=predictor, value=arclength)

        elif predictor == 'arclength_eigenvector':
            arclength = options['nonlinear/predictor/increment_length']
            model.predict_tangential(strategy='arc-length', value=arclength)
            beta = options['nonlinear/predictor/beta']
            model.combine_prediction_with_eigenvector(beta)

        else:
            raise Exception('Unknown predictor {}'.format(predictor))

        constraint = self.constraints[options['nonlinear/constraint_idx']]
        constraint_dof = model.free_dofs[options['nonlinear/constraint/dof_idx']]
        tolerance_power = options['nonlinear/newtonraphson/tolerance_power']
        max_iterations = options['nonlinear/newtonraphson/maxiterations']
        determinant = options['nonlinear/solution/det(K)_flag']
        eigenproblem = options['nonlinear/solution/eigenproblem_flag']

        model.perform_non_linear_solution_step(
            strategy=constraint,
            tolerance=10**tolerance_power,
            dof=constraint_dof,
            max_iterations=max_iterations,
            solve_det_k=determinant,
            solve_attendant_eigenvalue=eigenproblem)

        return model

    def solve_bracketing(self, model, options):
        backup_model = model.get_duplicate(branch=True)

        try:
            model = bracketing(
                model,
                tol=10** options['bracketing/tolerance_power'],
                max_steps=options['bracketing/maxiterations'],
                raise_error=True,
                tolerance=10**options['nonlinear/newtonraphson/tolerance_power'],
                max_iterations=options['nonlinear/newtonraphson/maxiterations'],
                solve_det_k=options['nonlinear/solution/det(K)_flag'],
                solve_attendant_eigenvalue=options['nonlinear/solution/eigenproblem_flag']
            )
        except Exception as e:
            traceback.print_exc()
            self.show_error_dialog(str(e))
            return backup_model

        return model

    def go_back_click(self):
        if self.model.get_previous_model() is None:
            return
        self.model = self.model.get_previous_model()
        self.redraw()
        self.DEBUG_red('Went back one step!')

    def reset_path_click(self):
        if self.model.get_previous_model() is None:
            return
        self.model = self.model.get_initial_model()
        self.redraw()
        self.DEBUG_red('Path has been reset!')

    def new_path_click(self):
        new_model = self.model.get_duplicate(branch=True)
        self.branches.append(new_model)
        self.redraw()
        self.DEBUG_blue('Created a new path!')

    def reset_all_click(self):
        model = self.model.get_initial_model()
        self.branches = [model]
        self.redraw()
        self.DEBUG_red('Reset all has been clicked!')


# == Loggers to be used in plots
class LoadDisplacementLogger(object):
    def __init__(self, dof):
        self.dof = dof
    @property
    def title(self):
        node_id, dof_type = self.dof
        return f'Load-displacement diagram for {dof_type} at node {node_id}'
    @property
    def xlabel(self):
        node_id, dof_type = self.dof
        return f'{dof_type} at node {node_id}'
    @property
    def ylabel(self):
        return 'Load factor (\u03BB)'
    def __call__(self, model):
        u = model.get_dof_state(self.dof)
        return model.get_dof_state(self.dof), model.lam

class CustomLogger(object):
    def __init__(self, x_fct, y_fct, x_label, y_label):
        self.x_fct = x_fct
        self.y_fct = y_fct
        self.xlabel = x_label
        self.ylabel = y_label
    @property
    def title(self):
        return '{} : {}'.format(self.xlabel, self.ylabel)
    def __call__(self, model):
        return self.x_fct(model), self.y_fct(model)
