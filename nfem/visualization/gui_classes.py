"""
Helper classes to be used in interactive.py
"""

import numpy as np

from nfem.visualization.python_ui import Widget, Figure, FigureCanvasQTAgg, NavigationToolbar2QT, QtWidgets
from nfem.visualization.plot import plot_model, plot_bounding_cube, plot_history_curve, plot_crosshair, plot_scaled_model, get_bounding_box
from nfem.visualization.plot import animate_model
from nfem.assembler import Assembler


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
                'Last Increment',
                'Arclength + Eigenvector'],
            option=builder.context.options['nonlinear/predictor_idx'])
        builder.add_stack(
            items=[
                SetLoadFactorPredictorSettings,
                IncrementLoadFactorPredictorSettings,
                SetDofValuePredictorSettings,
                IncrementDofValuePredictorSettings,
                ArclengthPredictorSettings,
                LastIncrementPredictorSettings,
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
        dofs = builder.context.model.dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in dofs]
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
        dofs = builder.context.model.dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in dofs]
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


class LastIncrementPredictorSettings(Widget):
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
        dofs = builder.context.model.dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in dofs]
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
        builder.add_button(
            label='Show Stiffness Matrix',
            action=self.show_stiffness_matrix)

    def show_stiffness_matrix(self, main_window):
        main_window.show_dialog(StiffnessMatrixDialog, title='Stiffness Matrix')


class StiffnessMatrixDialog(Widget):
    def build(self, builder):
        element_ids = ['Element '+str(element.id) for element in builder.context.model.elements]
        systems = ['Total System']
        systems.extend(element_ids)
        builder.add_combobox(
            label='System',
            items=systems,
            option=builder.context.options['stiffness/system_idx'])
        builder.add_space()
        builder.add_combobox(
            label='Stiffness Matrix Component',
            items=['Total', 'Elastic', 'Initial Displacement', 'Geometric'],
            option=builder.context.options['stiffness/component_idx'])
        builder.add_space()
        set_stiffness_matrix(
            builder.context.model,
            builder.context.DEBUG_blue,
            **builder.context.options
            )
        builder.add_array(
            readonly=True,
            option=builder.context.options['stiffness/matrix'])


def set_stiffness_matrix(model, debugger=print, **options):
    if options['stiffness/system_idx'].value == 0:

        assembler = Assembler(model)
        k = np.zeros((assembler.dof_count, assembler.dof_count))

        if options['stiffness/component_idx'].value == 0:
            assembler.assemble_matrix(k, lambda element: element.calculate_stiffness_matrix())
            debugger('Total stiffness matrix for the total system:')
        elif options['stiffness/component_idx'].value == 1:
            assembler.assemble_matrix(k, lambda element: element.calculate_elastic_stiffness_matrix())
            debugger('Elastic stiffness matrix for the total system:')
        elif options['stiffness/component_idx'].value == 2:
            assembler.assemble_matrix(k, lambda element: element.calculate_initial_displacement_stiffness_matrix())
            debugger('Initial displacement stiffness matrix for the total system:')
        elif options['stiffness/component_idx'].value == 3:
            assembler.assemble_matrix(k, lambda element: element.calculate_geometric_stiffness_matrix())
            debugger('Geometric stiffness matrix for the total system:')
        else:
            raise NotImplementedError(f'Wrong stiffness matrix component index.')

        options['stiffness/matrix'].change(k)
        debugger(str(k) + '\n')

    else:
        element = model.elements[options['stiffness/system_idx'].value - 1]

        if options['stiffness/component_idx'].value == 0:
            k = element.calculate_stiffness_matrix()
            debugger(f'Total stiffness matrix for element ({element.id}):')
        elif options['stiffness/component_idx'].value == 1:
            k = element.calculate_elastic_stiffness_matrix()
            debugger(f'Elastc stiffness matrix for element ({element.id}):')
        elif options['stiffness/component_idx'].value == 2:
            k = element.calculate_initial_displacement_stiffness_matrix()
            debugger(f'Initial displacement stiffness matrix for element ({element.id}):')
        elif options['stiffness/component_idx'].value == 3:
            k = element.calculate_geometric_stiffness_matrix()
            debugger(f'Geometric stiffness matrix for element ({element.id}):')
        else:
            raise NotImplementedError(f'Wrong stiffness matrix component index.')

        options['stiffness/matrix'].change(k)
        debugger(str(k) + '\n')


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
        dofs = builder.context.model.dofs
        dof_strings = [dof[1] + ' at node ' + str(dof[0]) for dof in dofs]

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
    def __init__(self):
        super(AnimationWindow, self).__init__()
        self.a = None   # Animation function return

    def build(self, builder):
        figure = Figure(dpi=80)
        animation_canvas = FigureCanvas(figure)
        animation_canvas.setContentsMargins(0, 0, 0, 0)
        builder.add(animation_canvas)
        ax_3d = figure.add_subplot(111, projection='3d')
        figure.tight_layout()
        ax_3d.set_aspect('equal')
        self.a = animate_model(
            figure, ax_3d,
            builder.context.model.get_model_history(),
            **builder.context.option_values
        )
        self.show()


class FigureCanvas(FigureCanvasQTAgg):
    """
    subclass of FigureCanvasQTAgg to be able to use
    python_ui.WidgetBuilder.add(widget_type)
    """
    def __call__(self):
        return self

    def build(self, builder):
        pass


# == content of the application window
class SideBySide2D3DPlots(QtWidgets.QWidget):
    def __init__(self, parent):
        super(SideBySide2D3DPlots, self).__init__()

        self.parent = parent

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
        toolbar3d.setMinimumWidth(self.width() / 2)
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
        toolbar2d.setMinimumWidth(self.width() / 2)
        layout.addWidget(toolbar2d, 2, 2, 1, 1)

        plot2d = figure2d.add_subplot(111)
        # plot2d.set_aspect('equal')
        self._plot2d = plot2d

    def redraw(self):
        plot3d = self._plot3d
        plot2d = self._plot2d

        figure3d = plot3d.get_figure()
        figure2d = plot2d.get_figure()
        figure3d.subplots_adjust()
        figure2d.subplots_adjust()

        plot3d.clear()
        plot2d.clear()

        plot2d.grid()

        self.plot(plot3d, plot2d)

        handles, labels = plot2d.get_legend_handles_labels()
        if handles:
            plot2d.legend(handles, labels, loc='upper right')

        self._canvas3d.draw()
        self._canvas2d.draw()

    def plot(self, ax3d, ax2d):
        # get variables
        parent = self.parent
        options = parent.option_values
        dof = parent.model.dofs[options['plot/dof_idx']]

        # plot bounding cube
        bounding_box = get_bounding_box(parent.model.get_model_history())
        plot_bounding_cube(ax3d, bounding_box)

        # plot initial and deformed models
        plot_model(ax3d, parent.model, 'gray', True, **options)
        plot_model(ax3d, parent.model, 'red', False, **options)

        # plot eigenvector
        if options['plot/eigenvector_flag'] and parent.model.first_eigenvector_model is not None:
            plot_scaled_model(ax3d, parent.model.first_eigenvector_model, 'green')

        # logger
        logger = LoadDisplacementLogger(dof)
        label = logger.xlabel + " : " + logger.ylabel
        ax2d.set(xlabel=logger.xlabel, ylabel=logger.ylabel, title=logger.title)
        ax2d.yaxis.set_label_position('right')
        ax3d.set(
            xlabel='< x >',
            ylabel='< y >',
            zlabel='< z >',
        )

        # plot load displacement curve
        if options['plot/load_disp_curve_flag']:
            # other branches at first level
            n_branches = len(parent.branches)
            for i, branch_model in enumerate(parent.branches[:-1]):
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
                model=parent.model,
                xy_function=logger,
                fmt='-o',
                label=label,
                color='tab:blue')
            plot_crosshair(
                ax=ax2d,
                x=parent.model[dof].delta,
                y=parent.model.load_factor,
                linestyle='-.',
                color='tab:blue',
                linewidth=0.75)

        # load displacement iteration plot
        if options['plot/load_disp_curve_iter_flag']:
            plot_history_curve(
                ax=ax2d,
                model=parent.model,
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
                x_fct=lambda model: model[dof].delta,
                y_fct=lambda model: model.det_k,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Det(K)')
            plot_history_curve(
                ax=ax2d,
                model=parent.model,
                xy_function=logger,
                fmt='-o',
                label=logger.title,
                color='tab:green')

        # eigenvalue plot
        if options['plot/eigenvalue_flag']:
            logger = CustomLogger(
                x_fct=lambda model: model[dof].delta,
                y_fct=lambda model: None if not model.first_eigenvalue else model.first_eigenvalue*model.load_factor,
                x_label=f'{dof[1]} at node {dof[0]}',
                y_label='Eigenvalue')
            plot_history_curve(
                ax=ax2d,
                model=parent.model,
                xy_function=logger,
                fmt='-o',
                label=logger.title,
                color='tab:red')


# == Loggers to be used in plots
class LoadDisplacementLogger:
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
        return model[self.dof].delta, model.load_factor


class CustomLogger:
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
