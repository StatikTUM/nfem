# PythonUI by Thomas Oberbichler
# https://github.com/oberbichler/PythonUI

import matplotlib
matplotlib.use("Qt5Agg") 
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
import inspect
import numpy as np
import sys


class _GenericValidator(QtGui.QValidator):
    def __init__(self, parent, validate):
        super(_GenericValidator, self).__init__(parent)
        self._validate = validate

    def validate(self, string, pos):
        validated = self._validate(string)

        if validated is None:
            return QtGui.QValidator.Invalid, string, pos
        elif validated == string:
            return QtGui.QValidator.Acceptable, string, pos
        else:
            return QtGui.QValidator.Intermediate, string, pos

    def fixup(self, string):
        return self._validate(string)


class Option(QtCore.QObject):
    _changed = QtCore.pyqtSignal

    def __init__(self, value, action=None):
        super(Option, self).__init__()
        self.value = value

        if action:
            self.connect(action)

    def connect(self, action):
        self._changed.connect(action)

    def change(self, value):
        self.value = value

    def __call__(self):
        return self.value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.emit()

    def emit(self):
        self._changed.emit(self._value)


class Fore:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[39m'


class Back:
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    RESET = '\033[49m'


class Style:
    DIM = '\033[2m'
    NORMAL = '\033[22m'
    BRIGHT = '\033[1m'
    RESET_ALL = '\033[0m'


ConsoleStyles = Fore, Back, Style


class DarkStyle:
    BACKGROUND = QtGui.QColor('#1b1b1b')
    BLACK = QtGui.QColor('#303030')
    RED = QtGui.QColor('#e1321a')
    GREEN = QtGui.QColor('#6ab017')
    YELLOW = QtGui.QColor('#ffc005')
    BLUE = QtGui.QColor('#004f9e')
    MAGENTA = QtGui.QColor('#ec0048')
    CYAN = QtGui.QColor('#2aa7e7')
    WHITE = QtGui.QColor('#f2f2f2')


class Console(QtWidgets.QTextEdit):

    def __init__(self):
        super(Console, self).__init__()
        
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        
        self.style = DarkStyle

        font = QtGui.QFont('Consolas')
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.setFont(font)
        self.setReadOnly(True)
        self.setFrameStyle(QtWidgets.QFrame.NoFrame)
        self._default_format = QtGui.QTextCharFormat()
        self._text_format = QtGui.QTextCharFormat()
        self.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        p = self.palette()
        p.setColor(QtGui.QPalette.Base, self.style.BACKGROUND)
        p.setColor(QtGui.QPalette.Text, self.style.WHITE)
        self.setPalette(p)

    def _foreground(self, color):
        if color is None:
            color = self._default_format.foreground()
        self._text_format.setForeground(color)

    def _background(self, color):
        if color is None:
            color = self._default_format.background()
        self._text_format.setBackground(color)

    def _apply_code(self, code):
        if code == 0:
            self._text_format.setFontWeight(self._text_format.fontWeight())
            self._foreground(None)
            self._background(None)
        elif code == 1:
            self._text_format.setFontWeight(QtGui.QFont.Bold)
        elif code == 30:
            self._foreground(self.style.BLACK)
        elif code == 31:
            self._foreground(self.style.RED)
        elif code == 32:
            self._foreground(self.style.GREEN)
        elif code == 33:
            self._foreground(self.style.YELLOW)
        elif code == 34:
            self._foreground(self.style.BLUE)
        elif code == 35:
            self._foreground(self.style.MAGENTA)
        elif code == 36:
            self._foreground(self.style.CYAN)
        elif code == 37:
            self._foreground(self.style.WHITE)
        elif code == 39:
            self._foreground(None)
        elif code == 40:
            self._background(self.style.BLACK)
        elif code == 41:
            self._background(self.style.RED)
        elif code == 42:
            self._background(self.style.GREEN)
        elif code == 43:
            self._background(self.style.YELLOW)
        elif code == 44:
            self._background(self.style.BLUE)
        elif code == 45:
            self._background(self.style.MAGENTA)
        elif code == 46:
            self._background(self.style.CYAN)
        elif code == 47:
            self._background(self.style.WHITE)
        elif code == 49:
            self._background(None)

    def write(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)

        npos = 0

        while True:
            start = text.find('\033[', npos)

            if start < 0:
                cursor.insertText(text[npos:], self._text_format)
                break

            if start != npos:
                cursor.insertText(text[npos:start], self._text_format)

            end = text.find('m', start + 2)

            for code in map(int, text[start+2:end].split(';')):
                self._apply_code(code)

            npos = end + 1

        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def flush(self):
        pass


class WidgetBuilder:
    def __init__(self, ground, context):
        self._ground = ground
        self.context = context

    def _add_widget(self, widget):
        self._ground.addWidget(widget)

    def add(self, widget_type):
        widget = widget_type()
        builder = WidgetBuilder(self._ground, self.context)
        widget.build(builder)
        self._add_widget(widget)

    def add_space(self):
        spacer_item = QtWidgets.QSpacerItem(16, 16)
        self._ground.addItem(spacer_item)

    def add_stretch(self):
        self._ground.addStretch(1)

    def add_label(self, label):
        label_widget = QtWidgets.QLabel(label)
        self._add_widget(label_widget)

    def add_button(self, label, action):
        button_widget = QtWidgets.QPushButton(label)
        self._add_widget(button_widget)

        if len(inspect.signature(action).parameters) == 0:
            button_widget.clicked.connect(action)
        else:
            button_widget.clicked.connect(lambda: action(self.context))

    def add_textbox(self, label, option, prefix=None, postfix=None,
                    validate=None, readonly=False):
        if label:
            label_widget = QtWidgets.QLabel(label)
            self._add_widget(label_widget)

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_widget.setLayout(row_layout)
        self._add_widget(row_widget)

        if prefix:
            prefix_widget = QtWidgets.QLabel(prefix)
            row_layout.addWidget(prefix_widget)

        textbox_widget = QtWidgets.QLineEdit()
        if validate:
            validator = _GenericValidator(textbox_widget, validate)
            textbox_widget.setValidator(validator)
        textbox_widget.setText(str(option.value))
        textbox_widget.setReadOnly(readonly)

        row_layout.addWidget(textbox_widget, 1)

        if option:
            option.connect(lambda value: textbox_widget.setText(str(value)))
            textbox_widget.editingFinished.connect(
                lambda: option.change(textbox_widget.text()))

        if postfix:
            postfix_widget = QtWidgets.QLabel(postfix)
            row_layout.addWidget(postfix_widget)

    def add_spinbox(self, label, option, prefix=None, postfix=None,
                    dtype=float, minimum=None, maximum=None, step=None,
                    decimals=None):
        if label:
            label_widget = QtWidgets.QLabel(label)
            self._add_widget(label_widget)

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_widget.setLayout(row_layout)
        self._add_widget(row_widget)

        if prefix:
            prefix_widget = QtWidgets.QLabel(prefix)
            row_layout.addWidget(prefix_widget)

        if dtype is int:
            spinbox_widget = QtWidgets.QSpinBox()
            spinbox_widget.setMinimum(minimum if minimum is not None else -2147483648)
            spinbox_widget.setMaximum(maximum if maximum is not None else 2147483647)
            spinbox_widget.setSingleStep(step or 1)
            spinbox_widget.setValue(option.value)
        elif dtype is float:
            spinbox_widget = QtWidgets.QDoubleSpinBox()
            spinbox_widget.setMinimum(minimum if minimum is not None else -Qt.qInf())
            spinbox_widget.setMaximum(maximum if maximum is not None else Qt.qInf())
            spinbox_widget.setSingleStep(step or 0.1)
            spinbox_widget.setDecimals(decimals or 5)
            spinbox_widget.setValue(option.value)
        else:
            raise ValueError(f'Invalid dtype "{dtype.__name__}"')

        spinbox_widget.setKeyboardTracking(False)
        row_layout.addWidget(spinbox_widget, 1)

        if option:
            option.connect(spinbox_widget.setValue)
            spinbox_widget.valueChanged.connect(option.change)

        if postfix:
            postfix_widget = QtWidgets.QLabel(postfix)
            row_layout.addWidget(postfix_widget)

    def add_checkbox(self, label, option):
        checkbox_widget = QtWidgets.QCheckBox(label)
        checkbox_widget.setChecked(option.value)
        self._add_widget(checkbox_widget)

        option.connect(checkbox_widget.setChecked)
        checkbox_widget.clicked.connect(option.change)

    def add_combobox(self, items, option, label=None):
        if label:
            label_widget = QtWidgets.QLabel(label)
            self._add_widget(label_widget)

        combobox_widget = QtWidgets.QComboBox()
        self._add_widget(combobox_widget)

        for item in items:
            combobox_widget.addItem(str(item))

        if option:
            option.connect(combobox_widget.setCurrentIndex)
            combobox_widget.currentIndexChanged.connect(option.change)
            combobox_widget.setCurrentIndex(option.value)

    def add_radiobuttons(self, items, option):
        button_group = QtWidgets.QButtonGroup(self._ground.parent())

        for i, item in enumerate(items):
            radio_button = QtWidgets.QRadioButton(item)
            button_group.addButton(radio_button, i)
            self._add_widget(radio_button)

        def toggled(button, checked):
            if not checked:
                return
            button_id = button_group.id(button)
            option.change(button_id)

        def toggle(button_id):
            button = button_group.button(button_id)
            button.toggle()

        if option:
            toggle(option.value)
            button_group.buttonToggled.connect(toggled)
            option.connect(toggle)

    def add_group(self, label, content):
        group_widget = QtWidgets.QGroupBox(label)
        self._add_widget(group_widget)

        group_layout = QtWidgets.QVBoxLayout()
        group_widget.setLayout(group_layout)

        content_widget = content()
        content_widget._build(self.context)

        group_layout.addWidget(content_widget)

    def add_tabs(self, items, option=None):
        tabs_widget = TabsWidget(self.context)
        self._add_widget(tabs_widget)

        for label, widget_type in items:
            tabs_widget.add_tab(label, widget_type)

        if option:
            option.connect(tabs_widget.setCurrentIndex)
            tabs_widget.currentChanged.connect(option.change)

    def add_stack(self, items, option=None):
        stack_widget = StackWidget(self.context, items, option)
        self._add_widget(stack_widget)

    def add_pages(self, items, option=None):
        pages_widget = PagesWidget(self.context, items, option)
        self._add_widget(pages_widget)

    def add_array(self, option, label=None, readonly=False):
        if label:
            label_widget = QtWidgets.QLabel(label)
            self._add_widget(label_widget)

        table_widget = QtWidgets.QTableWidget()
        if readonly:
            table_widget.setEditTriggers(
                QtWidgets.QAbstractItemView.NoEditTriggers)

        def update_table(array):
            table_widget.blockSignals(True)

            shape = np.shape(array)

            if len(shape) == 1:
                rows, = shape
                table_widget.setRowCount(rows)
                table_widget.setColumnCount(1)

                table_widget.setVerticalHeaderLabels(map(str, range(rows)))
                table_widget.setHorizontalHeaderLabels(['0'])

                for (i,), value in np.ndenumerate(array):
                    value_str = str(value)
                    table_widget.setItem(i, 0,
                                         QtWidgets.QTableWidgetItem(value_str))
            elif len(shape) == 2:
                rows, cols = shape
                table_widget.setRowCount(rows)
                table_widget.setColumnCount(cols)

                table_widget.setVerticalHeaderLabels(map(str, range(rows)))
                table_widget.setHorizontalHeaderLabels(map(str, range(cols)))

                for (i, j), value in np.ndenumerate(array):
                    value_str = str(value)
                    table_widget.setItem(i, j,
                                         QtWidgets.QTableWidgetItem(value_str))
            else:
                raise Exception('Arrays with dimension > 2 not supported')

            table_widget.blockSignals(False)

        def update_cell(row, col):
            shape = np.shape(option.value)

            value = float(table_widget.item(row, col).text())

            if len(shape) == 1:
                option.value[row] = value
            elif len(shape) == 2:
                option.value[row, col] = value
            else:
                raise Exception('Arrays with dimension > 2 not supported')

            option.emit()

        update_table(option.value)

        option.connect(update_table)
        table_widget.cellChanged.connect(update_cell)

        self._add_widget(table_widget)

    """
    contributed by Mahmoud Zidan
    """
    def add_slider(self, label, option, minimum=None, maximum=None, ticks=None):
        if label:
            label_widget = QtWidgets.QLabel(label)
            self._add_widget(label_widget)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(minimum or 1)
        slider.setMaximum(maximum or 10)
        if ticks:
            slider.setTickInterval(ticks)
            slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setValue(option.value)
        slider.valueChanged.connect(option.change)
        self._add_widget(slider)

    def add_wheel(self, option, unit=1):
        wheel = Wheel(option, unit)
        self._add_widget(wheel)

class Widget(QtWidgets.QWidget):
    def __init__(self):
        super(Widget, self).__init__()

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._ground = layout

    def _build(self, context):
        builder = WidgetBuilder(self._ground, context)
        self.build(builder)

    def build(self, builder):
        pass


class StackWidget(QtWidgets.QWidget):
    def __init__(self, context, items, option):
        super(StackWidget, self).__init__()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        stack = QtWidgets.QStackedWidget()
        layout.addWidget(stack)
        self._stack = stack

        for widget_type in items:
            widget = widget_type()

            builder = WidgetBuilder(widget._ground, context)
            widget.build(builder)

            if stack.count() != 0:
                widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                     QtWidgets.QSizePolicy.Ignored)

            stack.addWidget(widget)

        if option:
            option.connect(self._select_index)

        self._select_index(0)

    def _select_index(self, index):
        if self._stack.currentWidget():
            self._stack.currentWidget().setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Ignored)

        self._stack.setCurrentIndex(index)

        if self._stack.currentWidget():
            self._stack.currentWidget().setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred)


class PagesWidget(QtWidgets.QWidget):
    def __init__(self, context, pages, option):
        super(PagesWidget, self).__init__()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        combobox = QtWidgets.QComboBox()
        layout.addWidget(combobox)
        self._combobox = combobox

        stack = QtWidgets.QStackedWidget()
        layout.addWidget(stack)
        self._stack = stack

        for label, widget_type in pages:
            content = widget_type()

            builder = WidgetBuilder(content._ground, context)
            content.build(builder)

            self._add_page(label, content)

        combobox.currentIndexChanged.connect(self._select_index)

        if option:
            combobox.currentIndexChanged.connect(option.change)
            option.connect(self._select_index)

        self._select_index(0)

    def _select_index(self, index):
        self._combobox.setCurrentIndex(index)

        if self._stack.currentWidget():
            self._stack.currentWidget().setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Ignored)

        self._stack.setCurrentIndex(index)

        if self._stack.currentWidget():
            self._stack.currentWidget().setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred)

    def _add_page(self, key, widget):
        if self._stack.count() != 0:
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Ignored)

        self._combobox.addItem(str(key), key)

        self._stack.addWidget(widget)

        return widget


class TabsWidget(QtWidgets.QTabWidget):
    def __init__(self, context):
        super(TabsWidget, self).__init__()
        self.context = context

    def add_tab(self, label, widget_type=None):
        widget = widget_type()

        builder = WidgetBuilder(widget._ground, self.context)
        widget.build(builder)

        widget.setContentsMargins(8, 8, 8, 8)

        self.addTab(widget, label)

        return widget


class Sidebar(QtWidgets.QScrollArea):
    def __init__(self):
        super(Sidebar, self).__init__()

        widget = Widget()

        self._ground = widget._ground

        self.setWidget(widget)
        self._ground.setContentsMargins(8, 8, 8, 8)
        self.horizontalScrollBar().setEnabled(False)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setWidgetResizable(True)
        self.setMinimumWidth(350)


class PlotCanvas(QtWidgets.QWidget):
    def __init__(self, parent):
        super(PlotCanvas, self).__init__()

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        figure = Figure()
        canvas = FigureCanvasQTAgg(figure)
        canvas.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas, 1, 1, 1, 1)
        self._canvas = canvas

        toolbar = NavigationToolbar2QT(canvas, self)
        layout.addWidget(toolbar, 2, 1, 1, 1)

        plot = figure.add_subplot(111)
        plot.set_aspect('equal')
        self._plot = plot

    def redraw(self):
        plot = self._plot

        figure = plot.get_figure()

        for ax in figure.axes[1:]:
            figure.delaxes(ax)

        figure.subplots_adjust()

        plot.clear()

        self.plot(plot)

    def plot(self, ax):
        pass


class Wheel(QtWidgets.QDial):
    def __init__(self, option, unit=1):
        super().__init__()
        self.setMaximum(359)
        self.setValue(180)
        self._value = 180
        self.unit = unit
        self.valueChanged.connect(self._valueChanged)            
        self.setWrapping(True)
        self.option = option

    def _valueChanged(self, value):
        old_angle = self._value / 180 * np.pi
        new_angle = value / 180 * np.pi

        delta = np.arccos(np.cos(old_angle - new_angle))
        s = np.sign(-np.sin(old_angle - new_angle))

        self._value = int(self._value + s * delta * 180 / np.pi)

        self.option.value = self._value / 180 * self.unit


class ApplicationWindow(QtWidgets.QWidget):
    def __init__(self, title='', size=(1200, 800), content=None):
        super(ApplicationWindow, self).__init__()

        self.resize(*size)
        self.setWindowTitle(title)

        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.content = content(parent=self) if content else None
        self.console = Console()
        self.sidebar = Sidebar()

        if content:
            vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            vsplitter.addWidget(self.content)
            vsplitter.addWidget(self.console)
            vsplitter.setStretchFactor(0, 1)
            vsplitter.setStretchFactor(1, 0)

        hsplitter = QtWidgets.QSplitter()
        hsplitter.addWidget(self.sidebar)
        if content:
            hsplitter.addWidget(vsplitter)
        else:
            hsplitter.addWidget(self.console)
        hsplitter.setStretchFactor(0, 0)
        hsplitter.setStretchFactor(1, 1)

        self.layout().addWidget(hsplitter)

    def closeEvent(self, event):
        QtWidgets.QApplication.quit()

    @classmethod
    def run(cls, *args, **kwargs):
        app = QtWidgets.QApplication([])

        app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

        widget = cls(*args, **kwargs)
        widget._build(widget)

        widget.show()

        widget._started()

        app.exec_()

        return widget

    def show_dialog(self, widget_type, title='', size=None, modal=True,
                    action=None):
        dialog = QtWidgets.QDialog(self, QtCore.Qt.Tool)
        dialog.setWindowTitle(title)
        dialog.setModal(modal)

        widget = widget_type()
        widget._build(self)

        if size:
            dialog.resize(*size)
        else:
            dialog.adjustSize()

        layout = QtWidgets.QGridLayout()
        dialog.setLayout(layout)
        layout.addWidget(widget)

        def close_dialog(message=None):
            dialog.close()
            if action:
                action(message)

        self.close_dialog = close_dialog

        dialog.show()

    def show_open_file_dialog(self, title=None, extension_filters=None):
        if isinstance(extension_filters, list):
            extension_filter = ';;'.join(extension_filters)
        else:
            extension_filter = extension_filters

        result = QtWidgets.QFileDialog.getOpenFileName(self, title,
                                                       filter=extension_filter)

        return result

    def show_save_file_dialog(self, title=None, extension_filters=None):
        if isinstance(extension_filters, list):
            extension_filter = ';;'.join(extension_filters)
        else:
            extension_filter = extension_filters

        result = QtWidgets.QFileDialog.getSaveFileName(self, title,
                                                       filter=extension_filter)

        return result

    def show_error_dialog(self, message):
        QtWidgets.QMessageBox.critical(self, 'Error', message)

    def showEvent(self, event):
        self._old_stdout = sys.stdout
        sys.stdout = self.console

    def hideEvent(self, event):
        sys.stdout = self._old_stdout

    def _build(self, context):
        sidebar_builder = WidgetBuilder(self.sidebar._ground, context)
        self._build_sidebar(sidebar_builder)

    def _build_sidebar(self, builder):
        pass

    def _started(self):
        pass
