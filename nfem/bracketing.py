"""This file contains some the bracketing function to find the next critical point"""

import numpy as np

def bracketing(model, tol=1e-7, max_steps=100, **options):
    """Finds next critical point

    The bracketing function finds the next critical point starting from a models
    state.
    - A first nonlinear step needs to be solved already.
    - Det(K) has to be solved for the model.

    1. The algorithm searches for a sign change of Det(K).
        This is done by using a 'last-increment-prediction' and 'arc-length-control'
    2. A bisection algorithm is used to find the Det(K)=0 point.
        Again 'arc-length-control' strategy is used.

    Parameters
    ----------
    model : Model
        initial model for the bracketing function. A first step needs to be already solved
    tol : float
        tolerance for Det(K)==0
    max_steps : int
        maximum number of steps for the iterative search
    options :
        additional options for the nonlinear solution
    """

    print("=================================")
    print("Starting bracketing to search for next critical point.")

    if model.det_k == None:
        raise RuntimeError('Det(K) needs to be solved for the model.')

    if 'solve_attendant_eigenvalue' in options:
        solve_attendant_eigenvalue = options['solve_attendant_eigenvalue']
    else:
        solve_attendant_eigenvalue = True

    initial_model = model

    # search for the first sign change of det(K)
    success = False
    step = 0
    while step < max_steps and not success:

        step += 1

        model = model.get_duplicate()

        model.predict_tangential(strategy="arc-length")

        model.perform_non_linear_solution_step(strategy='arc-length-control',
                                            solve_det_k=True,
                                            solve_attendant_eigenvalue=solve_attendant_eigenvalue,
                                            **options)

        if np.sign(model.det_k) != np.sign(initial_model.det_k):
            success = True

    if success:
        print("=================================")
        print("Found switch in sign of Det(K) after {} steps.".format(step))
        print("=================================")
    else:
        raise RuntimeError('Bracketing was not succesful after {} steps.'.format(max_steps))

    # iterate until abs(det(K)) is < tol using bisection algorithm
    success = False
    lower_limit_model = model.get_previous_model()
    upper_limit_model = model
    while step < max_steps and not success:
        step += 1

        tmp_model = lower_limit_model.get_duplicate()

        for node in tmp_model.nodes:
            lower_node = lower_limit_model.get_node(id=node.id)
            upper_node = upper_limit_model.get_node(id=node.id)

            node.u = (lower_node.u + upper_node.u)/2
            node.v = (lower_node.v + upper_node.v)/2
            node.w = (lower_node.w + upper_node.w)/2

        tmp_model.lam = (lower_limit_model.lam + upper_limit_model.lam)/2

        tmp_model.perform_non_linear_solution_step(strategy='arc-length-control',
                                            solve_det_k=True,
                                            solve_attendant_eigenvalue=solve_attendant_eigenvalue,
                                            **options)

        if abs(tmp_model.det_k) < tol:
            tmp_model._previous_model = model.get_previous_model()
            success = True
            break

        if np.sign(lower_limit_model.det_k) == np.sign(tmp_model.det_k):
            lower_limit_model = tmp_model
        else:
            upper_limit_model = tmp_model

    if success:
        print("=================================")
        print("Found critical point after {} steps, Det(K) = {}.".format(step, tmp_model.det_k))
        print("=================================")
    else:
        raise RuntimeError('Bracketing was not succesful after {} steps.'.format(max_steps))

    return tmp_model

