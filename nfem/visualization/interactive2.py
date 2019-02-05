""" FIXME """

from .python_ui import ApplicationWindow, Option, Widget, Fore, Back, Style, QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits import mplot3d

def interact2(model, dof):
    """ FIXME """
    window = MainWindow.run(model=model, dof=dof)
    return window.model


class MainWindow(ApplicationWindow):
    """ FIXME """
    def __init__(self, model, dof):
        super(MainWindow, self).__init__(title='NFEM Teaching Tool')

        self.model = model
        self.dof = dof

        self.content.deleteLater()
        self.content = SideBySide2D3DPlots(self._draw)
        self.vsplitter.addWidget(self.content)
        self.vsplitter.addWidget(self.console)

        # == analysis options
        self.tab_idx = Option(0)
        self.solver_idx = Option(0)
        self.predictor_idx = Option(0)
        self.tangential_direction_chbx = Option(True)
        self.predictor_length = Option(0)
        self.constraint_idx = Option(0)
        self.maximum_iterations = Option(10)
        self.tolerance_power = Option(7)
        self.detK_chbx = Option(False)
        self.solve_eigenvalues_chbx = Option(False)

        # == visualization options
        self.highlight_dof_chbx = Option(True, self.redraw)
        self.show_dirichlet_chbx = Option(True, self.redraw)
        self.show_neumann_chbx = Option(True, self.redraw)
        self.symbol_size = Option(5, self.redraw)
        self.show_eigenvector_chbx = Option(False, self.redraw)
        self.load_disp_curve_chbx = Option(True, self.redraw)
        self.load_disp_curve_iter_chbx = Option(False, self.redraw)
        self.show_detK_chbx = Option(False, self.redraw)
        self.eigenvalue_chbx = Option(False, self.redraw)

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
                option=self.tab_idx)
        builder.add_stretch()
        builder.add_button(
            label='Solve',
            action=lambda: None)
        builder.add_button(
            label='Go back',
            action=lambda: None)
        builder.add_button(
            label='Reset path',
            action=lambda: None)
        builder.add_button(
            label='New path',
            action=lambda: None)
        builder.add_button(
            label='Reset all',
            action=lambda: None)

    def _draw(self, ax3d, ax2d):
        pass


class AnalysisTab(Widget):
    def build(self, builder):
        builder.add_combobox(
            label='Solver',
            items=[
                'Non-Linear',
                'Linearized Pre-Buckling (LPB)',
                'Linear',
                'Bracketing'],
            option=builder.context.solver_idx)
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
            option=builder.context.predictor_idx)
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.tangential_direction_chbx)
        builder.add_spinbox(
            label=None,
            option=builder.context.predictor_length)


class ConstraintGroup(Widget):
    def build(self, builder):
        builder.add_combobox(
            label=None,
            items=[
                'Load control',
                'Displacement control',
                'Arclength'],
            option=builder.context.constraint_idx)


class NewtonRaphsonGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label='Maximum Iterations',
            option=builder.context.maximum_iterations)
        builder.add_spinbox(
            label='Tolerance',
            prefix='10 ^ ',
            option=builder.context.tolerance_power)


class SolutionGroup(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.detK_chbx)
        builder.add_checkbox(
            label='Solve attendant eigenvalue analysis',
            option=builder.context.solve_eigenvalues_chbx)


class VisualisationTab(Widget):
    def build(self, builder):
        builder.add_group(
            label='3D Plot Settings',
            content=Plot3DSettingsGroup)
        builder.add_group(
            label='2D Plot Settings',
            content=Plot2DSettingsGroup)


class Plot3DSettingsGroup(Widget):
    def build(self, builder):
        builder.add_checkbox(
            label='Highlight Dof',
            option=builder.context.highlight_dof_chbx)
        builder.add_checkbox(
            label='Show Dirichlet BCs',
            option=builder.context.show_dirichlet_chbx)
        builder.add_checkbox(
            label='Show Neumann BCs',
            option=builder.context.show_neumann_chbx)
        builder.add_slider(
            label=None,
            option=builder.context.symbol_size)
        builder.add_checkbox(
            label='Show Eigenvector',
            option=builder.context.show_eigenvector_chbx)


class Plot2DSettingsGroup(Widget):
    def build(self, builder):
        # builder.add_group(
        #     )
        builder.add_checkbox(
            label='Load Displacement Curve',
            option=builder.context.load_disp_curve_chbx)
        builder.add_checkbox(
            label='Load Displacement Curve with iterations',
            option=builder.context.load_disp_curve_iter_chbx)
        builder.add_checkbox(
            label='Det(K)',
            option=builder.context.show_detK_chbx)
        builder.add_checkbox(
            label='Eigenvalue',
            option=builder.context.eigenvalue_chbx)


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
        # plot3d.set_aspect('equal')
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
        # plot2d.set_aspect('equal')
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
