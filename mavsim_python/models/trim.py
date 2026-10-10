"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion
from message_types.msg_delta import MsgDelta

def compute_trim(mav, Va, gamma, residual_tolerance=1e-2):
    """Find a zero-wind straight-flight trim, or raise on an unusable result.

    The zero-roll, zero-sideslip constraints can leave a small lateral residual
    because of propeller reaction torque. The default accepts a derivative
    residual norm up to 0.01; callers can request a tighter tolerance.
    """
    if not np.isfinite(Va) or Va <= 0:
        raise ValueError("Trim airspeed must be finite and positive.")
    if not np.isfinite(gamma) or abs(gamma) >= np.pi / 2:
        raise ValueError("Straight-flight trim requires |gamma| < pi/2.")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("Trim residual tolerance must be finite and positive.")

    # set the initial conditions of the optimization
    e0 = euler_to_quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # pn
                   [0],  # pe
                   [0],  # pd
                   [Va],  # u (initially aligned with the body x axis)
                   [0.], # v
                   [0.],  # w
                   [e0.item(0)],  # e0
                   [e0.item(1)],  # e1
                   [e0.item(2)],  # e2
                   [e0.item(3)],  # e3
                   [0.], # p
                   [0.], # q
                   [0.]  # r
                   ])
    delta0 = np.array([[-0.1],  # elevator
                       [0.0],  # aileron
                       [0.0],  # rudder
                       [0.5]]) # throttle
    x0 = np.concatenate((state0, delta0), axis=0)
    x0 = x0.flatten()  # scipy.optimize.minimize requires a 1-D initial guess
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    # Preserve the caller's state if the requested trim is infeasible.
    previous_state = mav._state.copy()
    previous_air_data = (mav._wind.copy(), mav._Va, mav._alpha, mav._beta)
    bounds = [(None, None)] * 16 + [(0.0, 1.0)]
    try:
        res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                       constraints=cons, bounds=bounds,
                       options={'ftol': 1e-10, 'maxiter': 500, 'disp': True})
        if not res.success or not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"Trim optimization failed: {res.message}")
        residual_norm = np.sqrt(trim_objective_fun(res.x, mav, Va, gamma))
        constraint_error = np.max(np.abs(cons['fun'](res.x)))
        if (not np.isfinite(residual_norm)
                or residual_norm > residual_tolerance
                or not np.isfinite(constraint_error)
                or constraint_error > 1e-6
                or not 0.0 <= res.x[16] <= 1.0):
            raise RuntimeError(
                f"No usable trim: derivative residual {residual_norm:.6g} "
                f"(limit {residual_tolerance:.6g}), "
                f"constraint error {constraint_error:.6g}.")
    except Exception:
        mav._state = previous_state
        mav._wind, mav._Va, mav._alpha, mav._beta = previous_air_data
        raise
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    mav._state = trim_state.copy()
    mav._update_velocity_data()
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized
    state = np.asarray(x[0:13]).reshape(13, 1)
    delta = MsgDelta(elevator=x.item(13),
                     aileron=x.item(14),
                     rudder=x.item(15),
                     throttle=x.item(16))
    desired_trim_state_dot = np.array([[0., 0., -Va*np.sin(gamma), 0.,
                                       0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    f = mav._f(state, forces_moments)
    tmp = desired_trim_state_dot - f
    J = np.linalg.norm(tmp[2:13])**2.0
    return J
