""" FIXME """

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits import mplot3d
import traceback

from .python_ui import ApplicationWindow, Option, Widget, Fore, Back, Style, QtWidgets, QtCore
from ..assembler import Assembler
from .plot import (plot_model, plot_load_displacement_curve, plot_bounding_cube,
                   plot_history_curve, plot_crosshair, plot_scaled_model)
from .plot import animate_model, get_bounding_box

def interact2(model, dof):
    """ FIXME """
    window = MainWindow.run(model=model, dof=dof)
    return window.model


class MainWindow(ApplicationWindow):
    def __init__(self, model, dof):
        super(MainWindow, self).__init__(
            title='NFEM Teaching Tool',
            content=SideBySide2D3DPlots(self._draw))

        self.branches = [model]
        self.dof = dof

        self.options = dict()
        # == analysis options
        self.options['tab_idx'] = Option(0)
        self.options['solver_idx'] = Option(0)

        self.options['linear/lambda'] = Option(0)

        self.options['nonlinear/predictor_idx'] = Option(0)
        self.options['nonlinear/predictor/tangential_flag'] = Option(True)
        self.options['nonlinear/predictor/lambda'] = Option(0.0)
        self.options['nonlinear/predictor/delta_lambda'] = Option(0.0)
        self.options['nonlinear/predictor/dof_idx'] = Option(0)
        self.options['nonlinear/predictor/dof_value'] = Option(0.0)
        self.options['nonlinear/predictor/increment_length'] = Option(0.0)
        self.options['nonlinear/predictor/beta'] = Option(0.0)

        self.options['nonlinear/constraint_idx'] = Option(0)
        self.options['nonlinear/constraint/dof_idx'] = Option(0)

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
        self.options['plot/dof_idx'] = Option(0, self.redraw) #TODO: make use of self.dof
        self.options['plot/load_disp_curve_flag'] = Option(True, self.redraw)
        self.options['plot/load_disp_curve_iter_flag'] = Option(False, self.redraw)
        self.options['plot/det(K)_flag'] = Option(False, self.redraw)
        self.options['plot/eigenvalue_flag'] = Option(False, self.redraw)

    @property
    def option_values(self):
        return {option[0] : option[1].value for option in self.options.items()}

    def get_solver(self):
        if self.options['solver_idx'].value == 0:
            return 'nonlinear'
        elif self.options['solver_idx'].value == 1:
            return 'LPB'
        elif self.options['solver_idx'].value == 2:
            return 'linear'
        elif self.options['solver_idx'].value == 3:
            return 'bracketing'
    
    def get_predictor(self):
        predictor_idx = self.options['nonlinear/predictor_idx'].value
        if predictor_idx == 0:
            return 'lambda'
        elif predictor_idx == 1:
            return 'delta_lambda'
        elif predictor_idx == 2:
            return 'dof'
        elif predictor_idx == 3:
            return 'delta_dof'
        elif predictor_idx == 4:
            return 'arclength'
        elif predictor_idx == 5:
            return 'arclength+eigenvector'
        
    def get_constraint(self):
        constraint_idx = self.options['nonlinear/constraint_idx'].value
        if constraint_idx == 0:
            return 'load-control'
        elif constraint_idx == 1:
            return 'displacement-control'
        elif constraint_idx == 2:
            return 'arc-length-control'
    
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
        model = self.model
        options = self.option_values
        free_dofs = Assembler(model).free_dofs
        dof = free_dofs[options['plot/dof_idx']]

        bounding_box = get_bounding_box(self.model.get_model_history())
        plot_bounding_cube(ax3d, bounding_box)

        plot_model(ax3d, self.model, 'gray', True, **options)
        plot_model(ax3d, self.model, 'red', False, **options)

        if options['plot/eigenvector_flag'] and model.first_eigenvector_model is not None:
            plot_scaled_model(ax3d, self.model.first_eigenvector_model, 'green')
        
        logger = LoadDisplacementLogger(dof)
        label = logger.xlabel + " : " + logger.ylabel
        if options['plot/load_disp_curve_flag']:
            # other branches at first level
            n_branches = len(self.branches)
            for i, branch_model in enumerate(self.branches[:-1]):
                grey_level = i/float(n_branches)
                plot_history_curve(ax2d, branch_model, logger, '--x', label=f'Branch {i+1} of {n_branches}', color=str(grey_level))
            # main branch
            plot_history_curve(ax2d, model, logger, '-o', label=label, color='tab:blue')
            plot_crosshair(ax2d, model.get_dof_state(dof), model.lam, linestyle='-.', color='tab:blue', linewidth=0.75)
        
        # load displacement iteration plot
        if options['plot/load_disp_curve_iter_flag']:
            plot_history_curve(ax2d, model, logger, '--o', label='{} (iter)'.format(label), skip_iterations=False, linewidth=0.75, markersize=2.0, color='tab:orange')

        # det_k plot
        if options['plot/det(K)_flag']:
            logger = CustomLogger(
                x_fct=lambda model: model.get_dof_state(dof=dof),
                y_fct=lambda model: model.det_k,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Det(K)')
            plot_history_curve(ax2d, model, logger, '-o', label=logger.title, color='tab:green')

        # eigenvalue plot
        if options['plot/eigenvalue_flag']:
            logger = CustomLogger(
                x_fct=lambda model: model.get_dof_state(dof=dof),
                y_fct=lambda model: None if not model.first_eigenvalue else model.first_eigenvalue*model.lam,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Eigenvalue')
            plot_history_curve(ax2d, model, logger, '-o', label=logger.title, color='tab:red')


    def solve_click(self):
        try:
            model = self.model.get_duplicate()
            options = self.option_values
            solver = self.get_solver()
            if solver == 'linear':
                model.lam = options['linear/lambda']
                model.perform_linear_solution_step()
            elif solver == 'LPB':
                if model.get_previous_model().get_previous_model() is not None:
                    raise RuntimeError('LPB can only be done on the initial model')
                model.lam = self.options['linear/lambda']
                model.perform_linear_solution_step()
                model.solve_linear_eigenvalues()
            elif solver == 'nonlinear':
                predictor = self.get_predictor()

                tangential_flag = options['nonlinear/predictor/tangential_flag']

                if predictor == 'lambda':
                    value = options['nonlinear/predictor/lambda']
                    if tangential_flag:
                        model.predict_tangential(strategy=predictor, value=value)
                    else:
                        model.lam = value
                elif predictor == 'delta_lambda':
                    value = options['nonlinear/predictor/delta_lambda']
                    if tangential_flag:
                        model.predict_tangential(strategy=predictor, value=value)
                    else:
                        model.lam += value
                elif predictor == 'dof':
                    dof = options['nonlinear/predictor/dof']
                    dof_value = options['nonlinear/predictor/dof_value']
                    if tangential_flag:
                        model.predict_tangential(strategy=predictor, dof=dof, value=dof_value)
                    else:
                        model.set_dof_state(dof, dof_value)
                elif predictor == 'delta_dof':
                    dof = options['nonlinear/predictor/dof']
                    dof_value_increment = options['nonlinear/predictor/delta_dof']
                    if tangential_flag:
                        model.predict_tangential(strategy=predictor, dof=dof, value=dof_value_increment)
                    else:
                        model.increment_dof_state(dof, dof_value_increment)
                elif predictor == 'arclength':
                    arclength = self.options['nonlinear/predictor/increment_length']
                    model.predict_tangential(strategy=predictor, value=arclength)
                # elif predictor == 'increment':
                #     increment_length = self.options['nonlinear/predictor/increment_length']
                #     model.predict_with_last_increment(value=increment_length)
                elif predictor == 'arclength_eigenvector':
                    arclength = options['nonlinear/predictor/increment_length']
                    model.predict_tangential(strategy='arc-length', value=arclength)
                    beta = options['nonlinear/predictor/beta']
                    model.combine_prediction_with_eigenvector(beta)
                else:
                    raise Exception('Unknown predictor {}'.format(predictor))

                constraint = self.get_constraint()
                constraint_dof = options['nonlinear/constraint/dof']
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
                    solve_attendant_eigenvalue=eigenproblem
                )
            elif solver == 'bracketing':
                raise NotImplementedError('Bracketing not yet implemented')
            else:
                raise Exception(f'Unknown solver {solver}')
        except Exception as e:
            traceback.print_exc()
            self.show_error_dialog(str(e))
            return
        self.model = model
        self.options['nonlinear/predictor/increment_length'].change(model.get_increment_norm())
        self.redraw()

    def go_back_click(self):
        self.DEBUG_red('go back is clicked')

    def reset_path_click(self):
        self.DEBUG_red('reset path is clicked')

    def new_path_click(self):
        self.DEBUG_blue('new path is clicked')

    def reset_all_click(self):
        self.WARNING('reset all is clicked')


class AnalysisTab(Widget):
    def build(self, builder):
        builder.add_combobox(
            label='Solver',
            items=[
                'Non-Linear',
                'Linearized Pre-Buckling (LPB)',
                'Linear',
                'Bracketing'],
            option=builder.context.options['solver_idx'])
        builder.add_stack(
            items=[
                NonLinearSettings,
                LPBSettings,
                LinearSettings,
                BracketingSettings],
            option=builder.context.options['solver_idx'])
        builder.add_stretch()


class NonLinearSettings(Widget):
    def build(self, builder):
        builder.add_group(
            label='Predictor',
            content=PredictorGroup)
        builder.add_group(
            label='Constraint',
            content=ConstraintGroup)
        builder.add_group(
            label='Newton-Raphson Settings',
            content=NewtonRaphsonGroup)
        builder.add_group(
            label='Solution',
            content=SolutionGroup)


class LPBSettings(Widget):
    def build(self, builder):
        builder.add_group(
            label='Load factor (\u03BB)',
            content=LinearLoadFactorGroup)


class LinearSettings(Widget):
    def build(self, builder):
        builder.add_group(
            label='Load factor (\u03BB)',
            content=LinearLoadFactorGroup)


class BracketingSettings(Widget):
    def build(self, builder):
        builder.add_group(
            label='Predefined Settings',
            content=PredefinedSettingsGroup)
        builder.add_group(
            label='Newton-Raphson Settings',
            content=NewtonRaphsonGroup)
        builder.add_group(
            label='Solution',
            content=SolutionGroup)
        builder.add_group(
            label='Bracketing',
            content=BracketingGroup)


class PredictorGroup(Widget):
    def build(self, builder):
        builder.add_combobox(
            label=None,
            items=[
                'Set load factor (\u03BB)',
                'Increment load factor (\u03BB)',
                'Set Dof value',
                'Increment Dof value',
                'Arclength',
                # 'Last increment',
                'Arclength + Eigenvector'],
            option=builder.context.options['nonlinear/predictor_idx'])
        builder.add_stack(
            items=[
                SetLoadFactorPredictorSettings,
                IncrementLoadFactorPredictorSettings,
                SetDofValuePredictorSettings,
                IncrementDofValuePredictorSettings,
                ArclengthPredictorSettings,
                # LastIncrement,
                ArclengthPlusEigenvectorPredictorSettings],
            option=builder.context.options['nonlinear/predictor_idx'])

class SetLoadFactorPredictorSettings(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['nonlinear/predictor/tangential_flag'])
        builder.add_spinbox(
            label=None,
            option=builder.context.options['nonlinear/predictor/lambda'])

class IncrementLoadFactorPredictorSettings(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['nonlinear/predictor/tangential_flag'])
        builder.add_spinbox(
            label=None,
            option=builder.context.options['nonlinear/predictor/delta_lambda'])

class SetDofValuePredictorSettings(Widget):
    def build(self, builder):
        # get the free dofs from the model
        assembler = Assembler(builder.context.model)
        dofs = [dof[1] + ' at node ' + dof[0] for dof in assembler.free_dofs]
        builder.add_combobox(
            items=dofs,
            option=builder.context.options['nonlinear/predictor/dof_idx'])
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['nonlinear/predictor/tangential_flag'])
        builder.add_spinbox(
            label=None,
            option=builder.context.options['nonlinear/predictor/dof_value'])

class IncrementDofValuePredictorSettings(Widget):
    def build(self, builder):
        # get the free dofs from the model
        assembler = Assembler(builder.context.model)
        dofs = [dof[1] + ' at node ' + dof[0] for dof in assembler.free_dofs]
        builder.add_combobox(
            items=dofs,
            option=builder.context.options['nonlinear/predictor/dof_idx'])
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['nonlinear/predictor/tangential_flag'])
        builder.add_spinbox(
            label=None,
            dtype=float,
            option=builder.context.options['nonlinear/predictor/increment_length'])

class ArclengthPredictorSettings(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label=None,
            dtype=float,
            option=builder.context.options['nonlinear/predictor/increment_length'])

class ArclengthPlusEigenvectorPredictorSettings(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label=None,
            dtype=float,
            option=builder.context.options['nonlinear/predictor/increment_length'])
        builder.add_spinbox(
            label='Beta',
            dtype=float,
            minimum=-1.0,
            maximum=1.0,
            option=builder.context.options['nonlinear/predictor/beta'])


class ConstraintGroup(Widget):
    def build(self, builder):
        builder.add_combobox(
            label=None,
            items=[
                'Load control',
                'Displacement control',
                'Arclength'],
            option=builder.context.options['nonlinear/constraint_idx'])
        builder.add_stack(
            items=[
                LoadControlConstraintSettings,
                DisplacementControlConstraintSettings,
                ArclengthConstaintSettings],
            option=builder.context.options['nonlinear/constraint_idx'])
class LoadControlConstraintSettings(Widget):
    def build(self, builder):
        pass
class ArclengthConstaintSettings(Widget):
    def build(self, builder):
        pass
class DisplacementControlConstraintSettings(Widget):
    def build(self, builder):
        # get the free dofs from the model
        assembler = Assembler(builder.context.model)
        dofs = [dof[1] + ' at node ' + dof[0] for dof in assembler.free_dofs]
        builder.add_combobox(
            items=dofs,
            option=builder.context.options['nonlinear/constraint/dof_idx'])



class NewtonRaphsonGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label='Maximum Iterations',
            option=builder.context.options['nonlinear/newtonraphson/maxiterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            option=builder.context.options['nonlinear/newtonraphson/tolerance_power'])


class SolutionGroup(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.options['nonlinear/solution/det(K)_flag'])
        builder.add_checkbox(
            label='Solve attendant eigenvalue analysis',
            option=builder.context.options['nonlinear/solution/eigenproblem_flag'])


class LinearLoadFactorGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label=None,
            option=builder.context.options['linear/lambda'])



class PredefinedSettingsGroup(Widget):
    def build(self, builder):
        builder.add_label(label="Predictor: Arclength")
        builder.add_label(label="Constraint: Arclength")



class BracketingGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label='Maximum iterations',
            option=builder.context.options['bracketing/maxiterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            option=builder.context.options['bracketing/tolerance_power'])



class VisualisationTab(Widget):
    def build(self, builder):
        builder.add_group(
            label='3D Plot Settings',
            content=Plot3DSettingsGroup)
        builder.add_group(
            label='2D Plot Settings',
            content=Plot2DSettingsGroup)
        builder.add_stretch()


class Plot3DSettingsGroup(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Highlight Dof',
            option=builder.context.options['plot/highlight_dof'])
        builder.add_checkbox(
            label='Show Dirichlet BCs',
            option=builder.context.options['plot/dirichlet'])
        builder.add_checkbox(
            label='Show Neumann BCs',
            option=builder.context.options['plot/neumann'])
        builder.add_slider(
            label=None,
            option=builder.context.options['plot/symbol_size'])
        builder.add_checkbox(
            label='Show Eigenvector',
            option=builder.context.options['plot/eigenvector_flag'])


class Plot2DSettingsGroup(Widget):
    def build(self, builder):
        # get the free dofs from the model
        assembler = Assembler(builder.context.model)
        dofs = [dof[1] + ' at node ' + dof[0] for dof in assembler.free_dofs]

        builder.add_combobox(
            items=dofs,
            option=builder.context.options['plot/dof_idx'])
        builder.add_checkbox(
            label='Load Displacement Curve',
            option=builder.context.options['plot/load_disp_curve_flag'])
        builder.add_checkbox(
            label='Load Displacement Curve with iterations',
            option=builder.context.options['plot/load_disp_curve_iter_flag'])
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.options['plot/det(K)_flag'])
        builder.add_checkbox(
            label='Eigenvalue',
            option=builder.context.options['plot/eigenvalue_flag'])


class SideBySide2D3DPlots(QtWidgets.QWidget):
    _redraw = QtCore.pyqtSignal(object, object)

    def __init__(self, redraw):
        super(SideBySide2D3DPlots, self).__init__()

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # left
        figure3d = Figure(dpi=80)
        canvas3d = FigureCanvasQTAgg(figure3d)
        canvas3d.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas3d, 1, 1, 1, 1)
        self._canvas3d = canvas3d

        toolbar3d = NavigationToolbar2QT(canvas3d, self)
        layout.addWidget(toolbar3d, 2, 1, 1, 1)

        plot3d = figure3d.add_subplot(111, projection='3d')
        plot3d.set_aspect('equal')
        self._plot3d = plot3d

        # right
        figure2d = Figure(dpi=80)
        canvas2d = FigureCanvasQTAgg(figure2d)
        canvas2d.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas2d, 1, 2, 1, 1)
        self._canvas2d = canvas2d

        toolbar2d = NavigationToolbar2QT(canvas2d, self)
        layout.addWidget(toolbar2d, 2, 2, 1, 1)

        plot2d = figure2d.add_subplot(111)
        plot2d.set_aspect('equal')
        self._plot2d = plot2d

        self._redraw.connect(redraw)

    def redraw(self):
        plot3d = self._plot3d
        plot2d = self._plot2d

        figure3d = plot3d.get_figure()
        figure2d = plot2d.get_figure()
        figure3d.subplots_adjust()
        figure2d.subplots_adjust()

        plot3d.clear()
        plot2d.clear()

        self._redraw.emit(plot3d, plot2d)

        plot2d.grid()
        handles, labels = plot2d.get_legend_handles_labels()
        if handles:
            plot2d.legend(handles, labels, loc='best')

        self._canvas3d.draw()
        self._canvas2d.draw()



# == Loggers
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
