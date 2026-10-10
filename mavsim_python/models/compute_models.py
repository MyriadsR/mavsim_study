"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion, quaternion_to_euler
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
from message_types.msg_delta import MsgDelta


def compute_model(mav, trim_state, trim_input):
    # Note: this function alters the mav private variables
    A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file (path relative to this file so the
    # launch file works from any working directory)
    file = open(Path(__file__).parent / 'model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.elevator, trim_input.aileron, trim_input.rudder, trim_input.throttle))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
     A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
     A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
     A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
     A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    file.write('B_lon = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lon[0][0], B_lon[0][1],
     B_lon[1][0], B_lon[1][1],
     B_lon[2][0], B_lon[2][1],
     B_lon[3][0], B_lon[3][1],
     B_lon[4][0], B_lon[4][1],))
    file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f],\n    '
               '[%f, %f, %f, %f, %f]])\n' %
    (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
     A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
     A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
     A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
     A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    file.write('B_lat = np.array([\n    [%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f],\n    '
               '[%f, %f]])\n' %
    (B_lat[0][0], B_lat[0][1],
     B_lat[1][0], B_lat[1][1],
     B_lat[2][0], B_lat[2][1],
     B_lat[3][0], B_lat[3][1],
     B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = quaternion_to_euler(trim_state[6:10])

    # define transfer function constants
    # dimensional stability derivatives from the Euler-state Jacobians
    # (u = [elevator, aileron, rudder, throttle], x_euler[9]=p, x_euler[10]=q)
    A = df_dx(mav, euler_state(trim_state), trim_input)
    B = df_du(mav, euler_state(trim_state), trim_input)
    # lateral: roll transfer function (slides 22, 24): a_phi1 = -L_p, a_phi2 = L_delta_a
    a_phi1 = -A.item(9, 9)
    a_phi2 = B.item(9, 1)
    # longitudinal: pitch transfer function (slides 29, 31): a_theta1 = -M_q, a_theta3 = M_delta_e
    a_theta1 = -A.item(10, 10)
    a_theta2 = -MAV.rho * Va_trim**2 * MAV.S_wing * MAV.c * MAV.C_m_alpha / (2 * MAV.Jy)
    a_theta3 = B.item(10, 0)

    # Compute transfer function coefficients using new propulsion model
    # airspeed transfer function (slides 35-36)
    a_V1 = (MAV.rho * Va_trim * MAV.S_wing / MAV.mass
            * (MAV.C_D_0 + MAV.C_D_alpha * alpha_trim
               + MAV.C_D_delta_e * trim_input.elevator)
            - dT_dVa(mav, Va_trim, trim_input.throttle) / MAV.mass)
    a_V2 = dT_ddelta_t(mav, Va_trim, trim_input.throttle) / MAV.mass
    a_V3 = MAV.gravity * np.cos(theta_trim - alpha_trim)

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3


def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)

    A = df_dx(mav, x_euler, trim_input)
    B = df_du(mav, x_euler, trim_input)
    # selection matrices (slides 53-54): x_euler = [pn,pe,pd,u,v,w,phi,theta,psi,p,q,r],
    # u = [elevator, aileron, rudder, throttle]
    E1 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # u
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # w
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # q
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # theta
                   [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # h = -pd
    E2 = np.array([[1, 0, 0, 0],   # elevator
                   [0, 0, 0, 1]])  # throttle
    E3 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # v
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # p
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # r
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # phi
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])  # psi
    E4 = np.array([[0, 1, 0, 0],   # aileron
                   [0, 0, 1, 0]])  # rudder
    # extract longitudinal states (u, w, q, theta, h)
    A_lon = E1 @ A @ E1.T
    B_lon = E1 @ B @ E2.T
    # extract lateral states (v, p, r, phi, psi)
    A_lat = E3 @ A @ E3.T
    B_lat = E3 @ B @ E4.T
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    phi, theta, psi = quaternion_to_euler(x_quat[6:10])
    x_euler = np.concatenate((x_quat[0:6],
                              np.array([[phi], [theta], [psi]]),
                              x_quat[10:13]), axis=0)
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    e = euler_to_quaternion(x_euler.item(6), x_euler.item(7), x_euler.item(8))
    x_quat = np.concatenate((x_euler[0:6], e, x_euler[9:12]), axis=0)
    return x_quat

def f_euler(mav, x_euler, delta):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state, f_euler will be f, except for the attitude states

    # need to correct attitude states by multiplying f by
    # partial of quaternion_to_euler(quat) with respect to quat
    # dEuler/dt = dEuler/dquat * dquat/dt
    x_quat = quaternion_state(x_euler)
    mav._state = x_quat
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    f = mav._f(x_quat, forces_moments)
    # attitude kinematics in Euler angles (slide 3)
    phi = x_euler.item(6)
    theta = x_euler.item(7)
    p = x_euler.item(9)
    q = x_euler.item(10)
    r = x_euler.item(11)
    f_euler_ = np.concatenate((
        f[0:6],
        np.array([[p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],
                  [q*np.cos(phi) - r*np.sin(phi)],
                  [(q*np.sin(phi) + r*np.cos(phi))/np.cos(theta)]]),
        f[10:13]), axis=0)
    return f_euler_

def df_dx(mav, x_euler, delta):
    # take partial of f_euler with respect to x_euler
    eps = 0.01  # deviation
    m = 12
    n = 12
    A = np.zeros((m, n))  # Jacobian of f wrt x
    f_at_x = f_euler(mav, x_euler, delta)
    for i in range(0, n):
        x_eps = np.copy(x_euler)
        x_eps[i][0] += eps  # add eps to ith state
        f_at_x_eps = f_euler(mav, x_eps, delta)
        df_dxi = (f_at_x_eps - f_at_x) / eps
        A[:, i] = df_dxi[:, 0]
    # restore the state of the mav at x_euler
    mav._state = quaternion_state(x_euler)
    return A


def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to input
    eps = 0.01  # deviation
    m = 12
    n = 4
    B = np.zeros((m, n))  # Jacobian of f wrt u
    f_at_x = f_euler(mav, x_euler, delta)
    input_list = ['elevator', 'aileron', 'rudder', 'throttle']
    for i in range(0, n):
        delta_eps = MsgDelta(elevator=delta.elevator,
                             aileron=delta.aileron,
                             rudder=delta.rudder,
                             throttle=delta.throttle)
        setattr(delta_eps, input_list[i], getattr(delta, input_list[i]) + eps)
        f_at_x_eps = f_euler(mav, x_euler, delta_eps)
        df_dui = (f_at_x_eps - f_at_x) / eps
        B[:, i] = df_dui[:, 0]
    # restore the state of the mav at x_euler
    mav._state = quaternion_state(x_euler)
    return B


def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    eps = 0.01
    T_at_Va = mav._motor_thrust_torque(Va, delta_t)[0]
    T_at_Va_eps = mav._motor_thrust_torque(Va + eps, delta_t)[0]
    dT_dVa = (T_at_Va_eps - T_at_Va) / eps
    return dT_dVa

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    eps = 0.01
    T_at_dt = mav._motor_thrust_torque(Va, delta_t)[0]
    T_at_dt_eps = mav._motor_thrust_torque(Va, delta_t + eps)[0]
    dT_ddelta_t = (T_at_dt_eps - T_at_dt) / eps
    return dT_ddelta_t
