"""This module sets the ipython environment such, that the nfem tool works properly.
If it is imported from within a standard python console, it has no effect.    
"""

try:
    from IPython import get_ipython

    # clear the ipython history
    get_ipython().magic('reset -sf')
    print("Executed IPython magic command: %'reset -sf' to delete the history.")
    
    # set the matplotlib to tk, so the plots show
    get_ipython().magic('matplotlib automatic')
    print("Executed IPython magic command: %'matplotlib automatic' to show.")

except:
    pass


