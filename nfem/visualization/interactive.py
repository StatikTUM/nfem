from tkinter import Tk
import tkinter.ttk as tk

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from .plot import *

from ..path_following_method import LoadControl, DisplacementControl, ArcLengthControl
from ..predictor import LoadIncrementPredictor, DisplacementIncrementPredictor

class InteractiveWindow(Tk):
    def __init__(self, model, dof, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        self.geometry('1000x400')
        self.title('NFEM Teaching Tool')

        self.branches = [model]
        self.dof = dof

        # --- sidebar

        sidebar = tk.Frame(self, width=200)
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        self.sidebar = sidebar

        button = tk.Button(sidebar, text='Load controlled step', command=self.load_control_button_click)
        button.pack(fill='x', padx=4, pady=(4, 2))

        button = tk.Button(sidebar, text='Displacement controlled step', command=self.displacement_control_button_click)
        button.pack(fill='x', padx=4, pady=2)

        button = tk.Button(sidebar, text='Arc-length controlled step', command=self.arc_length_control_button_click)
        button.pack(fill='x', padx=4, pady=2)

        button = tk.Button(sidebar, text='Reset', command=self.reset_button_click)
        button.pack(fill='x', side='bottom', padx=4, pady=(2, 4))

        button = tk.Button(sidebar, text='Go back', command=self.go_back_button_click)
        button.pack(fill='x', side='bottom', padx=4, pady=2)

        button = tk.Button(sidebar, text='New branch', command=self.new_branch_button_click)
        button.pack(fill='x', side='bottom', padx=4, pady=2)

        # --- plot_canvas

        figure = Figure(dpi=80)
        self.figure = figure

        plot_canvas = FigureCanvasTkAgg(figure, self)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(expand=True, fill='both', side='right')
        self.plot_canvas = plot_canvas

        plot_3d = figure.add_subplot(1, 2, 1, projection='3d')
        plot_3d.set_proj_type('ortho')
        self.plot_3d = plot_3d

        plot_2d = figure.add_subplot(1, 2, 2)
        self.plot_2d = plot_2d

        self.redraw()

    @property
    def model(self):
        return self.branches[-1]

    @model.setter
    def model(self, value):
        self.branches[-1] = value

    def load_control_button_click(self):
        model = self.model.get_duplicate()

        model.lam += 0.1

        predictor = LoadIncrementPredictor()

        method = LoadControl(model.lam)
        
        model.perform_non_linear_solution_step(predictor_method=predictor,
                                           path_following_method=method)

        self.model = model

        self.redraw()

    def displacement_control_button_click(self):
        model = self.model.get_duplicate()

        displacement = -0.1

        dof = self.dof

        predictor_method = DisplacementIncrementPredictor(dof=dof, value=displacement)

        displacement_hat = model.get_dof_state(dof) + displacement

        path_following_method = DisplacementControl(dof=dof, displacement_hat=displacement_hat)

        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

        self.model = model

        self.redraw()

    def arc_length_control_button_click(self):
        model = self.model.get_duplicate()

        dof = self.dof

        arclength = 0.12
        predictor_method = DisplacementIncrementPredictor(dof=dof, value=-1.0)
        path_following_method = ArcLengthControl(l_hat=arclength)
        
        model.perform_non_linear_solution_step(predictor_method=predictor_method,
                                           path_following_method=path_following_method)

        self.model = model

        self.redraw()

    def new_branch_button_click(self):
        new_model = self.model.get_duplicate()

        new_model.previous_model = self.model.previous_model

        self.branches.append(new_model)

        self.redraw()

    def go_back_button_click(self):
        if self.model.previous_model is None:
            return

        self.model = self.model.previous_model

        self.redraw()

    def reset_button_click(self):        
        model = self.model.get_initial_model()

        self.branches = [model]

        self.redraw()

    def redraw(self):
        model = self.model
        node_id, dof_type = self.dof

        plot_3d = self.plot_3d
        plot_2d = self.plot_2d
        
        plot_3d.clear()
        plot_3d.grid()

        min_x, max_x, min_y, max_y, min_z, max_z = bounding_box(model)
        
        plot_3d.set_xlim(min_x, max_x)
        plot_3d.set_ylim(min_y, max_y)
        plot_3d.set_zlim(min_z, max_z)

        plot_model(plot_3d, model, 'gray', True)

        plot_model(plot_3d, model, 'red', False)

        plot_2d.clear()
        plot_2d.set(xlabel='{} at node {}'.format(dof_type, node_id), ylabel='Load factor ($\lambda$)', title='Load-displacement diagram')
        plot_2d.set_facecolor('white')
        plot_2d.yaxis.tick_right()
        plot_2d.yaxis.set_label_position("right")
        plot_2d.grid()

        for model in self.branches:
            plot_load_displacement_curve(plot_2d, model, self.dof)

        self.plot_canvas.draw()

def interact(model, dof):
    window = InteractiveWindow(model, dof=('B', 'v'))
    window.mainloop()

    return window.model
