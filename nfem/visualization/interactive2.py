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

        self.spinbox_number = Option(1, self.redraw)

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
        builder.add_button(label='normal message', action=lambda: self.DEBUG('normal message'))
        builder.add_button(label='blue message', action=lambda: self.DEBUG_blue('blue message'))
        builder.add_button(label='red message', action=lambda: self.DEBUG_red('red message'))
        builder.add_button(label='warning', action=lambda: self.WARNING('warning string ....'))
        builder.add_stretch()
    
    def _draw(self, ax3d, ax2d):
        pass


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
        # canvas3d.setSizePolicy() #FIXME
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

        self._canvas3d.draw()
        self._canvas2d.draw()
