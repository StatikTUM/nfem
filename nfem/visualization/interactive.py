"""This module contains an interactive user interface.

Author: Thomas Oberbichler
"""


from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QPushButton, QSpinBox,
                             QStackedWidget, QTextEdit, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import sys

from .plot import (plot_model, plot_load_displacement_curve, plot_bounding_cube,
                   plot_history_curve)
from .plot import animate_model, get_bounding_box
from ..assembler import Assembler
from ..bracketing import bracketing

def interact(model, dof):
    app = QApplication([])

    window = InteractiveWindow(model, dof=dof)
    window.show()

    app.exec_()

    return window.model


class InteractiveWindow(QWidget):
    def __init__(self, model, dof):
        super(InteractiveWindow, self).__init__()

        self.options = Options()

        self.branches = [model]
        self.dof = dof

        self.animation_window = None

        # --- setup window

        self.resize(1000, 400)
        self.setWindowTitle('NFEM Teaching Tool')

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        # --- sidebar

        sidebar = Sidebar(self)
        layout.addWidget(sidebar, 1, 1, 2, 1)

        # --- plot_canvas

        canvas = Canvas(self)
        layout.addWidget(canvas, 1, 2, 1, 1)
        self.redraw = canvas.redraw

        # --- log

        widget = QTextEdit()
        widget.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        widget.setReadOnly(True)
        widget.setFixedHeight(200)
        widget.setFrameStyle(QFrame.HLine)
        layout.addWidget(widget, 2, 2, 1, 1)
        self.logTextEdit = widget

    @property
    def model(self):
        return self.branches[-1]

    @model.setter
    def model(self, value):
        self.branches[-1] = value

    def solve_click(self):
        try:
            model = self.model.get_duplicate()

            solver = self.options['solver']

            if solver == 'linear':
                model.lam = self.options['linear/loadfactor']
                model.perform_linear_solution_step()
            elif solver == 'nonlinear':
                predictor = self.options['nonlinear/predictor']

                if predictor == 'loadfactor':
                    model.lam = self.options['nonlinear/predictor/loadfactor']
                elif predictor == 'loadfactor_increment':
                    model.lam += self.options['nonlinear/predictor/loadfactor_increment']
                elif predictor == 'dof_value':
                    dof = self.options['nonlinear/predictor/dof']
                    dof_value = self.options['nonlinear/predictor/dof_value']
                    model.set_dof_state(dof, dof_value)
                    print(model.get_dof_state(dof))
                elif predictor == 'dof_value_increment':
                    dof = self.options['nonlinear/predictor/dof']
                    dof_value_increment = self.options['nonlinear/predictor/dof_value_increment']
                    model.increment_dof_state(dof, dof_value_increment)
                    print(model.get_dof_state(dof))
                elif predictor == 'arc-length':
                    arclength = self.options['nonlinear/predictor/increment_length']
                    model.predict_tangential(strategy=predictor, value=arclength)
                elif predictor == 'increment':
                    increment_length = self.options['nonlinear/predictor/increment_length']
                    model.predict_with_last_increment(value=increment_length)
                else:
                    raise Exception('Unknown predictor {}'.format(predictor))

                constraint = self.options['nonlinear/constraint']
                constraint_dof = self.options['nonlinear/constraint/dof']
                tolerance = self.options['nonlinear/newtonraphson/tolerance']
                max_iterations = self.options['nonlinear/newtonraphson/maxiterations']
                determinant = self.options['nonlinear/solution/determinant']
                eigenproblem = self.options['nonlinear/solution/eigenproblem']

                model.perform_non_linear_solution_step(
                    strategy=constraint,
                    tolerance=10**tolerance,
                    dof=constraint_dof,
                    max_iterations=max_iterations,
                    solve_det_k=determinant,
                    solve_attendant_eigenvalue=eigenproblem
                )
            else:
                raise Exception('Unknown solver {}'.format(solver))
        except Exception as e:
            QMessageBox(QMessageBox.Critical, 'Error', str(e), QMessageBox.Ok, self).show()
            return

        self.model = model

        self.options['nonlinear/predictor/increment_length'] = model.get_increment_norm()

        self.redraw()

    def bracketing_click(self):

        model = self.model
        backup_model = model.get_duplicate()
        backup_model._previous_model = model._previous_model

        try:
            tolerance = self.options['nonlinear/newtonraphson/tolerance']
            max_iterations = self.options['nonlinear/newtonraphson/maxiterations']

            determinant = self.options['nonlinear/solution/determinant']
            eigenproblem = self.options['nonlinear/solution/eigenproblem']

            bracketing_tolerance = self.options['nonlinear/bracketing/tolerance']
            bracketing_maxiterations = self.options['nonlinear/bracketing/maxiterations']

            model = bracketing(
                model,
                tol=10**bracketing_tolerance,
                max_steps=bracketing_maxiterations,
                raise_error=True,
                tolerance=10**tolerance,
                max_iterations=max_iterations,
                solve_det_k=determinant,
                solve_attendant_eigenvalue=eigenproblem
            )

        except Exception as e:
            QMessageBox(QMessageBox.Critical, 'Error', str(e), QMessageBox.Ok, self).show()
            model = backup_model
            return

        self.model = model

        self.redraw()

    def go_back_click(self):
        if self.model.get_previous_model() is None:
            return

        self.model = self.model.get_previous_model()

        self.redraw()

    def reset_path_click(self):
        if self.model.get_previous_model() is None:
            return

        self.model = self.model.get_initial_model()

        self.redraw()

    def new_path_click(self):
        new_model = self.model.get_duplicate()

        new_model._previous_model= self.model.get_previous_model()

        self.branches.append(new_model)

        self.redraw()

    def reset_all_click(self):
        model = self.model.get_initial_model()

        self.branches = [model]

        self.redraw()

    def show_animation_click(self):
        if self.animation_window is not None:
            self.animation_window.close()
        model = self.model
        self.animation_window = AnimationWindow(self, model)

    def showEvent(self, event):
        # redirect console output
        sys.stdout = Stream(textWritten=self.write_log)

    def closeEvent(self, event):
        # restore default console output
        sys.stdout = sys.__stdout__

        if self.animation_window is not None:
            self.animation_window.close()

        super(InteractiveWindow, self).closeEvent(event)

    def write_log(self, text):
        cursor = self.logTextEdit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.logTextEdit.setTextCursor(cursor)
        self.logTextEdit.ensureCursorVisible()

def _dof_to_str(dof):
    node_id, dof_type = dof
    return '\'{}\' at \'{}\''.format(dof_type, node_id)

def _free_dof_items(model):
    # FIXME remove this function
    assembler = Assembler(model)

    return [(_dof_to_str(dof), dof) for dof in assembler.free_dofs]


class LoadDisplacementLogger(object):
    def __init__(self, dof):
        self.dof = dof

    @property
    def title(self):
        node_id, dof_type = self.dof
        return 'Load-displacement diagram for {} at node {}'.format(dof_type, node_id)

    @property
    def xlabel(self):
        node_id, dof_type = self.dof
        return '{} at node {}'.format(dof_type, node_id)

    @property
    def ylabel(self):
        return 'Load factor ($\lambda$)'.format()

    def __call__(self, model):
        return model.get_dof_state(self.dof), model.lam


class Options(QObject):
    changed = pyqtSignal(str)

    def __init__(self):
        super(Options, self).__init__()
        self._root = ''
        self._data = dict()

        # linear
        self['linear/loadfactor'] = 1.0

        # predictor
        self['nonlinear/predictor/loadfactor'] = 0.0
        self['nonlinear/predictor/loadfactor_increment'] = 0.1
        self['nonlinear/predictor/dof_value'] = 0.0
        self['nonlinear/predictor/dof_value_increment'] = 0.1
        self['nonlinear/predictor/increment_length'] = 0.0

        # constraint
        self['nonlinear/constraint/dof'] = None

        # Newton-Raphson
        self['nonlinear/newtonraphson/maxiterations'] = 1000
        self['nonlinear/newtonraphson/tolerance'] = -5

        # solution
        self['nonlinear/solution/determinant'] = False
        self['nonlinear/solution/eigenproblem'] = False

        # bracketing
        self['nonlinear/bracketing/tolerance'] = -7
        self['nonlinear/bracketing/maxiterations'] = 100

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

        self.changed.emit(key)


class WidgetBase(QWidget):
    def __init__(self, parent):
        super(WidgetBase, self).__init__(parent)

    def master(self):
        current = self

        while current.parent() is not None:
            current = current.parent()

        return current

    def get_option(self, key):
        return self.master().options[key]

    def options(self):
        return self.master().options

    def set_option(self, key, value):
        self.master().options[key] = value

class Widget(WidgetBase):
    def __init__(self, parent, widgets=[]):
        super(Widget, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self._layout = layout

        for widget in widgets:
            self.add_widget(widget)

    def add_widget(self, widget):
        self._layout.addWidget(widget)

    def add_group(self, label, content=None):
        group = QGroupBox(label)
        self.add_widget(group)

        layout = QVBoxLayout()
        group.setLayout(layout)

        widget = content or Widget(self)

        layout.addWidget(widget)

        layout.addStretch(1)

        return widget

    def add_stack(self, *args, **kwargs):
        stack = StackWidget(self, *args, **kwargs)
        self.add_widget(stack)
        return stack

    def add_stretch(self):
        self._layout.addStretch(1)

    def add_spinbox(self, label=None, dtype=int, prefix=None, postfix=None, step=None, minimum=None, maximum=None, decimals=None, option_key=None):
        if label is not None:
            widget = QLabel(label)
            self.add_widget(widget)

        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row.setLayout(row_layout)
        self.add_widget(row)

        if prefix:
            widget = QLabel(prefix)
            row_layout.addWidget(widget)

        if dtype is int:
            widget = QSpinBox()
            widget.setMinimum(minimum or -100)
            widget.setMaximum(maximum or 100)
            widget.setSingleStep(step or 1)
        elif dtype is float:
            widget = QDoubleSpinBox()
            widget.setMinimum(minimum or -Qt.qInf())
            widget.setMaximum(maximum or Qt.qInf())
            widget.setSingleStep(step or 0.1)
            widget.setDecimals(decimals or 5)
        else:
            raise ValueError('Wrong dtype "{}"'.format(dtype.__name__))

        widget.setValue(self.get_option(option_key))
        widget.valueChanged.connect(lambda value: self.set_option(option_key, value))

        def on_options_changed(key):
            if key == option_key:
                widget.setValue(self.get_option(option_key))

        self.options().changed.connect(on_options_changed)

        row_layout.addWidget(widget, 1)

        if postfix:
            widget = QLabel(postfix)
            row_layout.addWidget(widget)

        return widget

    def add_checkbox(self, label, option_key):
        widget = QCheckBox(label)
        widget.setChecked(self.get_option(option_key))
        widget.stateChanged.connect(lambda value: self.set_option(option_key, value != 0))
        self.add_widget(widget)

    def add_combobox(self, items, option_key):
        widget = QComboBox()

        for label, value in items:
            widget.addItem(label, value)

        widget.currentIndexChanged.connect(lambda index: self.set_option(option_key, widget.currentData()))
        widget.setCurrentIndex(0)

        self.add_widget(widget)

    def add_free_dof_combobox(self, option_key):
        assembler = Assembler(self.master().model)

        return self.add_combobox(
            items=[(_dof_to_str(dof), dof) for dof in assembler.free_dofs],
            option_key=option_key
        )

    def add_button(self, label, action):
        button = QPushButton(label)
        button.clicked.connect(action)
        self._layout.addWidget(button)

        return button

class StackWidget(WidgetBase):
    def __init__(self, parent, option_key=None):
        super(StackWidget, self).__init__(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        combobox = QComboBox()
        layout.addWidget(combobox)
        self._combobox = combobox

        stack = QStackedWidget()
        layout.addWidget(stack)
        self._stack = stack

        combobox.currentIndexChanged.connect(stack.setCurrentIndex)
        combobox.currentIndexChanged.connect(lambda value: self.set_option(option_key, combobox.currentData()))

        layout.addStretch(1)

    def add_page(self, label, option_value=None, content=None):
        self._combobox.addItem(label, option_value)

        widget = content or Widget(self)
        self._stack.addWidget(widget)

        return widget

    def selected_widget(self):
        return self._stack.currentWidget()


class Sidebar(Widget):
    def __init__(self, parent):
        super(Sidebar, self).__init__(parent)

        self.setFixedWidth(250)
        self._layout.setContentsMargins(8, 8, 8, 8)

        stack = self.add_stack(option_key='solver')
        stack.add_page(
            label='Linear',
            option_value='linear',
            content=LinearSettings(self)
        )
        stack.add_page(
            label='Nonlinear',
            option_value='nonlinear',
            content=NonlinearSettings(self)
        )

        self.add_stretch()

        self.add_button(
            label='Solve',
            action=self.master().solve_click
        )
        self.add_button(
            label='Go back',
            action=self.master().go_back_click
        )
        self.add_button(
            label='Reset path',
            action=self.master().reset_path_click
        )
        self.add_button(
            label='New path',
            action=self.master().new_path_click
        )
        self.add_button(
            label='Reset all',
            action=self.master().reset_all_click
        )
        self.add_button(
            label='Show animation',
            action=self.master().show_animation_click
        )

class Canvas(WidgetBase):
    def __init__(self, parent):
        super(Canvas, self).__init__(parent)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        figure3d = Figure(dpi=80)
        canvas3d = FigureCanvasQTAgg(figure3d)
        canvas3d.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas3d, 1, 1, 2, 1)
        self.canvas3d = canvas3d

        plot3d = figure3d.add_subplot(111, projection='3d')
        self.plot3d = plot3d

        figure2d = Figure(dpi=80)
        canvas2d = FigureCanvasQTAgg(figure2d)
        canvas2d.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas2d, 2, 2, 1, 1)
        self.canvas2d = canvas2d

        plot2d = figure2d.add_subplot(111)
        self.plot2d = plot2d

        assembler = Assembler(self.master().model)

        # FIXME make this cleaner...
        dof_selector = QComboBox()
        for dof in assembler.free_dofs:
            logger = LoadDisplacementLogger(dof)
            dof_selector.addItem(logger.title, logger)
        layout.addWidget(dof_selector, 1, 2, 1, 1)
        dof_selector.currentIndexChanged.connect(lambda _: self.set_option('plot/logger', dof_selector.currentData()))
        dof_selector.currentIndexChanged.connect(lambda _: self.redraw())
        dof_selector.setCurrentIndex(1)
        dof_selector.setCurrentIndex(0)

        self.redraw()

    def redraw(self):
        model = self.master().model

        logger = self.get_option('plot/logger')

        node_id, dof_type = logger.dof

        plot3d = self.plot3d
        plot2d = self.plot2d

        plot3d.clear()
        plot3d.grid()

        bounding_box = get_bounding_box(model.get_model_history())
        plot_bounding_cube(plot3d, bounding_box)

        plot_model(plot3d, model, 'gray', True)
        plot_model(plot3d, model, 'red', False)

        plot2d.clear()
        #plot2d.set(xlabel=logger.xlabel, ylabel=logger.ylabel, title=logger.title)
        plot2d.set_facecolor('white')
        plot2d.yaxis.tick_right()
        #plot2d.yaxis.set_label_position('right')
        plot2d.grid()

        for model in self.master().branches:
            label = logger.xlabel + " : " + logger.ylabel
            plot_history_curve(plot2d, model, logger, '-o', label=label, skip_iterations=True)
            label += " (iter)"
            plot_history_curve(plot2d, model, logger, '--o', label=label, skip_iterations=False, linewidth=0.75, markersize=2.0)

        plot2d.legend(loc='best')

        self.canvas3d.draw()
        self.canvas2d.draw()


class LinearSettings(Widget):
    def __init__(self, parent):
        super(LinearSettings, self).__init__(parent)

        settings = self.add_group('Load factor (λ)')
        settings.add_spinbox(
            dtype=float,
            option_key='linear/loadfactor'
        )

        self.add_stretch()

class NonlinearSettings(Widget):
    def __init__(self, parent):
        super(NonlinearSettings, self).__init__(parent)

        self.add_group(
            label='Predictor',
            content=PredictorSettings(self)
        )
        self.add_group(
            label='Constraint',
            content=ConstraintSettings(self)
        )
        self.add_group(
            label='Newton-Raphson',
            content=NewtonRaphsonSettings(self)
        )
        self.add_group(
            label='Solution',
            content=SolutionSettings(self)
        )
        self.add_group(
            label='Bracketing',
            content=BracketingSettings(self)
        )

        self.add_stretch()


class PredictorSettings(StackWidget):
    def __init__(self, parent):
        super(PredictorSettings, self).__init__(parent,
            option_key='nonlinear/predictor'
        )

        self.add_page(
            label='Set load factor (λ)',
            option_value='loadfactor',
            content=LoadPredictorSettings(self)
        )
        self.add_page(
            label='Increment load factor (λ)',
            option_value='loadfactor_increment',
            content=LoadIncrementPredictorSettings(self)
        )
        self.add_page(
            label='Set Dof value',
            option_value='dof_value',
            content=DofPredictorSettings(self)
        )
        self.add_page(
            label='Increment Dof value',
            option_value='dof_value_increment',
            content=DofIncrementPredictorSettings(self)
        )
        self.add_page(
            label='Arclength',
            option_value='arc-length',
            content=ArclengthPredictorSettings(self)
        )
        self.add_page(
            label='Last Increment',
            option_value='increment',
            content=LastIncrementPredictorSettings(self)
        )

class ConstraintSettings(StackWidget):
    def __init__(self, parent):
        super(ConstraintSettings, self).__init__(parent,
            option_key='nonlinear/constraint'
        )

        self.add_page(
            label='Load control',
            option_value='load-control',
            content=LoadControlSettings(parent)
        )
        self.add_page(
            label='Displacement control',
            option_value='displacement-control',
            content=DisplacementControlSettings(parent)
        )
        self.add_page(
            label='Arc-length',
            option_value='arc-length-control',
            content=ArcLengthSettings(parent)
        )

class NewtonRaphsonSettings(Widget):
    def __init__(self, parent):
        super(NewtonRaphsonSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=int,
            label='Maximum iterations',
            minimum=1,
            maximum=5000,
            option_key='nonlinear/newtonraphson/maxiterations'
        )
        self.add_spinbox(
            dtype=int,
            label='Tolerance',
            prefix='10^',
            minimum=-10,
            maximum=-1,
            option_key='nonlinear/newtonraphson/tolerance'
        )

class SolutionSettings(Widget):
    def __init__(self, parent):
        super(SolutionSettings, self).__init__(parent)

        self.add_checkbox(
            label='Det(K)',
            option_key='nonlinear/solution/determinant'
        )
        self.add_checkbox(
            label='Solve attendant eigenvalue analysis',
            option_key='nonlinear/solution/eigenproblem'
        )


class BracketingSettings(Widget):
    def __init__(self, parent):
        super(BracketingSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=int,
            label='Maximum iterations',
            minimum=1,
            maximum=5000,
            option_key='nonlinear/bracketing/maxiterations'
        )
        self.add_spinbox(
            dtype=int,
            label='Tolerance',
            prefix='10^',
            minimum=-10,
            maximum=-1,
            option_key='nonlinear/bracketing/tolerance'
        )

        self.add_button(
            label="Solve Bracketing",
            action=self.master().bracketing_click
        )

class LoadPredictorSettings(Widget):
    def __init__(self, parent):
        super(LoadPredictorSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/loadfactor'
        )
        self.add_stretch()

class LoadIncrementPredictorSettings(Widget):
    def __init__(self, parent):
        super(LoadIncrementPredictorSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/loadfactor_increment'
        )
        self.add_stretch()

class DofPredictorSettings(Widget):
    def __init__(self, parent):
        super(DofPredictorSettings, self).__init__(parent)

        self.add_free_dof_combobox(
            option_key='nonlinear/predictor/dof'
        )

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/dof_value'
        )

        self.add_stretch()

class DofIncrementPredictorSettings(Widget):
    def __init__(self, parent):
        super(DofIncrementPredictorSettings, self).__init__(parent)

        self.add_free_dof_combobox(
            option_key='nonlinear/predictor/dof'
        )

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/dof_value_increment'
        )

        self.add_stretch()


class ArclengthPredictorSettings(Widget):
    def __init__(self, parent):
        super(ArclengthPredictorSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/increment_length'
        )

        self.add_stretch()

class LastIncrementPredictorSettings(Widget):
    def __init__(self, parent):
        super(LastIncrementPredictorSettings, self).__init__(parent)

        self.add_spinbox(
            dtype=float,
            option_key='nonlinear/predictor/increment_length'
        )

        self.add_stretch()


class LoadControlSettings(Widget):
    def __init__(self, parent):
        super(LoadControlSettings, self).__init__(parent)

        self.add_stretch()

class DisplacementControlSettings(Widget):
    def __init__(self, parent):
        super(DisplacementControlSettings, self).__init__(parent)

        self.add_free_dof_combobox(
            option_key='nonlinear/constraint/dof'
        )

        self.add_stretch()

class ArcLengthSettings(Widget):
    def __init__(self, parent):
        super(ArcLengthSettings, self).__init__(parent)

        self.add_stretch()


class AnimationWindow(QWidget):
    def __init__(self, parent, model):
        super(AnimationWindow, self).__init__()

        self.setWindowTitle('Animation')

        layout = QVBoxLayout()

        figure = Figure(dpi=80)
        figure = figure

        animation_canvas = FigureCanvasQTAgg(figure)
        animation_canvas.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(animation_canvas)

        ax_3d = figure.add_subplot(1, 1, 1, projection='3d')
        ax_3d = ax_3d

        self.setLayout(layout)

        # store the animation
        self.a = animate_model(figure, ax_3d, model.get_model_history())

        self.show()


class Stream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
