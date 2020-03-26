try:
    from nfem.visualization.interactive import interact # needs to be imported first, because it sets the matplotlib backend
except Exception as _:
    pass

try:
    get_ipython()
    from nfem.visualization.notebook_animation import show_load_displacement_curve, show_animation, show_deformation_plot
    from nfem.visualization.notebook_plot import Plot2D
except Exception as _:
    from nfem.visualization.plot import show_load_displacement_curve, show_animation, show_deformation_plot
    from nfem.visualization.plot import Animation3D, DeformationPlot3D
