""" FIXME """

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits import mplot3d

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
        self.options['predictor_idx'] = Option(0)
        self.options['tangential_direction_chbx'] = Option(True)
        self.options['nonlinear/predictor/lambda'] = Option(0.0)
        self.options['linear/lambda'] = Option(0)
        self.options['constraint_idx'] = Option(0)
        self.options['maximum_iterations'] = Option(100)
        self.options['tolerance_power'] = Option(-7)
        self.options['detK_chbx'] = Option(False)
        self.options['solve_eigenvalues_chbx'] = Option(False)
        self.options['bracketing_max_iterations'] = Option(100)
        self.options['bracketing_tolerance_power'] = Option(-7)

        # == visualization options
        self.options['plot/highlight_dof'] = Option(True, self.redraw)
        self.options['plot/dirichlet'] = Option(True, self.redraw)
        self.options['plot/neumann'] = Option(True, self.redraw)
        self.options['plot/symbol_size'] = Option(5, self.redraw)
        self.options['show_eigenvector_chbx'] = Option(False, self.redraw)
        self.options['free_dofs_idx'] = Option(0, self.redraw)
        self.options['load_disp_curve_chbx'] = Option(True, self.redraw)
        self.options['load_disp_curve_iter_chbx'] = Option(False, self.redraw)
        self.options['show_detK_chbx'] = Option(False, self.redraw)
        self.options['eigenvalue_chbx'] = Option(False, self.redraw)

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
        bounding_box = get_bounding_box(self.model.get_model_history())
        plot_bounding_cube(ax3d, bounding_box)

        plot_model(ax3d, self.model, 'gray', True, **self.options)
        plot_model(ax3d, self.model, 'red', False, **self.options)

        if self.options['show_eigenvector_chbx'].value and self.model.first_eigenvector_model is not None:
            plot_scaled_model(ax3d, self.model.first_eigenvector_model, 'green')
        
        #TODO: get the dof from self.free_dofs_idx

    def solve_click(self):
        self.DEBUG('solve is clicked')

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
                'Last increment',
                'Arclength + Eigenvector'],
            option=builder.context.options['predictor_idx'])
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['tangential_direction_chbx'])
        builder.add_spinbox(
            label=None,
            option=builder.context.options['nonlinear/predictor/lambda'])


class ConstraintGroup(Widget):
    def build(self, builder):
        builder.add_combobox(
            label=None,
            items=[
                'Load control',
                'Displacement control',
                'Arclength'],
            option=builder.context.options['constraint_idx'])


class NewtonRaphsonGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label='Maximum Iterations',
            option=builder.context.options['maximum_iterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            option=builder.context.options['tolerance_power'])


class SolutionGroup(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.options['detK_chbx'])
        builder.add_checkbox(
            label='Solve attendant eigenvalue analysis',
            option=builder.context.options['solve_eigenvalues_chbx'])


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
            option=builder.context.options['bracketing_max_iterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            option=builder.context.options['bracketing_tolerance_power'])



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
            option=builder.context.options['show_eigenvector_chbx'])


class Plot2DSettingsGroup(Widget):
    def build(self, builder):
        # get the free dofs from the model
        assembler = Assembler(builder.context.model)
        dofs = list()
        for dof in assembler.free_dofs:
            dofs.append(dof[1] + ' at node ' + dof[0])

        builder.add_combobox(
            items=dofs,
            option=builder.context.options['free_dofs_idx'])
        builder.add_checkbox(
            label='Load Displacement Curve',
            option=builder.context.options['load_disp_curve_chbx'])
        builder.add_checkbox(
            label='Load Displacement Curve with iterations',
            option=builder.context.options['load_disp_curve_iter_chbx'])
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.options['show_detK_chbx'])
        builder.add_checkbox(
            label='Eigenvalue',
            option=builder.context.options['eigenvalue_chbx'])


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
