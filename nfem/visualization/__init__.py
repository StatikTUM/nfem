try:
    get_ipython()
    _is_notebook = True
except Exception as _:
    _is_notebook = False

if _is_notebook:
    from nfem.visualization.notebook_animation import show_animation, show_deformation_plot
    from nfem.visualization.notebook_plot import show_load_displacement_curve, Plot2D
else:
    from nfem.visualization.plot import show_load_displacement_curve, show_animation, show_deformation_plot
    from nfem.visualization.plot import Plot2D
