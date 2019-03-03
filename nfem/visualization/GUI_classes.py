"""
Helper classes to be used in interactive.py
"""

from .python_ui import (Widget, Figure, FigureCanvasQTAgg, NavigationToolbar2QT,
                        QtWidgets, QtCore)
from .plot import animate_model

class AnalysisTab(Widget):
    def build(self, builder):
        builder.add_combobox(
            label='Solver',
            items=[
                'Nonlinear',
                'Linearized Pre-Buckling (LPB)',
                'Linear',
                'Bracketing'
            ],
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
        free_dofs = builder.context.model.free_dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in free_dofs]
        builder.add_combobox(
            items=dof_strings,
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
        free_dofs = builder.context.model.free_dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in free_dofs]
        builder.add_combobox(
            items=dof_strings,
            option=builder.context.options['nonlinear/predictor/dof_idx'])
        builder.add_checkbox(
            label='Tangential direction',
            option=builder.context.options['nonlinear/predictor/tangential_flag'])
        builder.add_spinbox(
            label=None,
            dtype=float,
            option=builder.context.options['nonlinear/predictor/delta-dof'])


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

class DisplacementControlConstraintSettings(Widget):
    def build(self, builder):
        # get the free dofs from the model
        free_dofs = builder.context.model.free_dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in free_dofs]
        builder.add_combobox(
            items=dof_strings,
            option=builder.context.options['nonlinear/constraint/dof_idx'])

class ArclengthConstaintSettings(Widget):
    def build(self, builder):
        pass



class NewtonRaphsonGroup(Widget):
    def build(self, builder):
        builder.add_spinbox(
            label='Maximum Iterations',
            dtype=int,
            minimum=1,
            maximum=5000,
            option=builder.context.options['nonlinear/newtonraphson/maxiterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            dtype=int,
            minimum=-10,
            maximum=-1,
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
            dtype=int,
            minimum=1,
            maximum=5000,
            option=builder.context.options['bracketing/maxiterations'])
        builder.add_spinbox(
            label='Tolerance',
            prefix='10^ ',
            dtype=int,
            minimum=-10,
            maximum=-1,
            option=builder.context.options['bracketing/tolerance_power'])



class VisualisationTab(Widget):
    def build(self, builder):
        builder.add_group(
            label='3D Plot Settings',
            content=Plot3DSettingsGroup)
        builder.add_group(
            label='2D Plot Settings',
            content=Plot2DSettingsGroup)
        builder.add_space()
        builder.add_button(
            label='Show Animation',
            action=builder.context.show_animation_click)
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
        free_dofs = builder.context.model.free_dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in free_dofs]

        builder.add_combobox(
            items=dof_strings,
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


class AnimationWindow(Widget):
    def build(self, builder):
        figure = Figure(dpi=80)
        animation_canvas = FigureCanvasQTAgg(figure)
        animation_canvas.setContentsMargins(0, 0, 0, 0)
        builder._add_widget(animation_canvas)
        ax_3d = figure.add_subplot(111, projection='3d')
        figure.tight_layout()
        ax_3d.set_aspect('equal')
        animation = animate_model(
            figure, ax_3d,
            builder.context.model.get_model_history(),
            **builder.context.option_values
        )
        self.show()


# == content of the application window
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
        toolbar3d.setMinimumSize(canvas3d.width(), 20)
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
        toolbar2d.setMinimumSize(canvas2d.width(), 20)
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
