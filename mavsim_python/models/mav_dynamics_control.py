"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # Controls and wind are held constant over this sample, but aerodynamic
        # forces must be recomputed at each RK4 stage's state.
        state = self._state[0:13].copy()
        dt = self._ts_simulation
        k1 = self._control_derivatives(state, delta, wind)
        k2 = self._control_derivatives(state + dt / 2 * k1, delta, wind)
        k3 = self._control_derivatives(state + dt / 2 * k2, delta, wind)
        k4 = self._control_derivatives(state + dt * k3, delta, wind)
        self._state[0:13] = state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self._state[6:10] /= np.linalg.norm(self._state[6:10])
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        self._forces = self._forces_moments(delta)[0:3].copy()
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _control_derivatives(self, state, delta, wind):
        self._state[0:13] = state
        self._update_velocity_data(wind)
        return self._f(state, self._forces_moments(delta))

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        # convert steady-state wind vector from world to body frame
        R_bi = quaternion_to_rotation(self._state[6:10])
        wind_body = R_bi.T @ steady_state
        # add the gust (defined in the body frame)
        wind_body = wind_body + gust
        # convert total wind to world frame
        self._wind = R_bi @ wind_body

        # velocity vector relative to the airmass
        ur = self._state.item(3) - wind_body.item(0)
        vr = self._state.item(4) - wind_body.item(1)
        wr = self._state.item(5) - wind_body.item(2)

        # compute airspeed
        self._Va = np.sqrt(ur**2.0 + vr**2.0 + wr**2.0)

        # compute angle of attack
        self._alpha = np.arctan2(wr, ur)

        # compute sideslip angle
        self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # extract states (phi, theta, psi, p, q, r)
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        fg_x = -MAV.mass * MAV.gravity * np.sin(theta)
        fg_y = MAV.mass * MAV.gravity * np.cos(theta) * np.sin(phi)
        fg_z = MAV.mass * MAV.gravity * np.cos(theta) * np.cos(phi)

        # compute Lift and Drag coefficients (CL, CD) - eqs. (4.9)-(4.11)
        # sigma(alpha) blends the linear lift curve into the post-stall regime;
        # the exponents are clipped to keep exp() finite at extreme alpha
        exp_lo = np.exp(np.clip(-MAV.M * (self._alpha - MAV.alpha0), -700.0, 700.0))
        exp_up = np.exp(np.clip( MAV.M * (self._alpha + MAV.alpha0), -700.0, 700.0))
        sigma = (1.0 + exp_lo + exp_up) / ((1.0 + exp_lo) * (1.0 + exp_up))
        CL_base = MAV.C_L_0 + MAV.C_L_alpha * self._alpha
        CL = (1.0 - sigma) * CL_base \
             + sigma * (2.0 * np.sign(self._alpha) * np.sin(self._alpha)**2.0 * np.cos(self._alpha)) \
             + MAV.C_L_q * MAV.c * q / (2.0 * self._Va) \
             + MAV.C_L_delta_e * delta.elevator
        # parasitic drag C_D_p plus induced drag proportional to the square of
        # the alpha-only lift coefficient, eq. (4.11)
        CD = MAV.C_D_p + CL_base**2.0 / (np.pi * MAV.e * MAV.AR) \
             + MAV.C_D_q * MAV.c * q / (2.0 * self._Va) \
             + MAV.C_D_delta_e * delta.elevator

        # compute Lift and Drag Forces (F_lift, F_drag)
        F_lift = 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * CL
        F_drag = 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * CD

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        # compute longitudinal forces in body frame (fx, fz)
        # lift/drag are defined in the stability frame and rotated into the
        # body frame by the angle-of-attack only; sideslip enters the lateral
        # dynamics through C_Y(beta)
        fx = -F_drag * np.cos(self._alpha) + F_lift * np.sin(self._alpha) + fg_x + thrust_prop
        fz = -F_drag * np.sin(self._alpha) - F_lift * np.cos(self._alpha) + fg_z

        # compute lateral forces in body frame (fy)
        # note: the drag side-force component -F_drag*sin(beta) is omitted to
        # match the chapter 4 reference values (the C_Y_beta term dominates)
        fy = fg_y \
             + 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * (
                 MAV.C_Y_0 + MAV.C_Y_beta * self._beta
                 + MAV.C_Y_p * MAV.b * p / (2.0 * self._Va)
                 + MAV.C_Y_r * MAV.b * r / (2.0 * self._Va)
                 + MAV.C_Y_delta_a * delta.aileron
                 + MAV.C_Y_delta_r * delta.rudder)

        # compute logitudinal torque in body frame (My)
        My = 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * MAV.c * (
             MAV.C_m_0 + MAV.C_m_alpha * self._alpha
             + MAV.C_m_q * MAV.c * q / (2.0 * self._Va)
             + MAV.C_m_delta_e * delta.elevator)

        # compute lateral torques in body frame (Mx, Mz)
        Mx = 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * MAV.b * (
             MAV.C_ell_0 + MAV.C_ell_beta * self._beta
             + MAV.C_ell_p * MAV.b * p / (2.0 * self._Va)
             + MAV.C_ell_r * MAV.b * r / (2.0 * self._Va)
             + MAV.C_ell_delta_a * delta.aileron
             + MAV.C_ell_delta_r * delta.rudder) - torque_prop
        Mz = 0.5 * MAV.rho * self._Va**2.0 * MAV.S_wing * MAV.b * (
             MAV.C_n_0 + MAV.C_n_beta * self._beta
             + MAV.C_n_p * MAV.b * p / (2.0 * self._Va)
             + MAV.C_n_r * MAV.b * r / (2.0 * self._Va)
             + MAV.C_n_delta_a * delta.aileron
             + MAV.C_n_delta_r * delta.rudder)

        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller (slides 34-36)
        # map delta_t throttle command(0 to 1) into motor input voltage
        v_in = delta_t * MAV.V_max

        # Angular speed of propeller: positive root of Q_motor(Q) = Q_prop(Q),
        #   a*Q^2 + b*Q + c = 0  (slide 35)
        #   Q_prop = rho*(Q/2pi)^2 D^5 C_Q(J) with J = 2*pi*Va/(Q*D)
        a = MAV.rho * MAV.D_prop**5.0 / (2.0 * np.pi)**2.0 * MAV.C_Q0
        b = MAV.rho * MAV.D_prop**4.0 / (2.0 * np.pi) * MAV.C_Q1 * Va \
            + MAV.KQ * MAV.KV / MAV.R_motor
        c = MAV.rho * MAV.D_prop**3.0 * MAV.C_Q2 * Va**2.0 \
            - MAV.KQ / MAV.R_motor * v_in + MAV.KQ * MAV.i0
        disc = b**2.0 - 4.0 * a * c
        if disc <= 0.0:
            omega = 0.0
        else:
            omega = (-b + np.sqrt(disc)) / (2.0 * a)
        if omega <= 0.0:
            omega = 0.0

        # thrust and torque due to propeller (slides 31-32, 36)
        if omega > 0.0:
            J = 2.0 * np.pi * Va / (omega * MAV.D_prop)
            C_T = MAV.C_T2 * J**2.0 + MAV.C_T1 * J + MAV.C_T0
            C_Q = MAV.C_Q2 * J**2.0 + MAV.C_Q1 * J + MAV.C_Q0
            thrust_prop = MAV.rho * (omega / (2.0 * np.pi))**2.0 * MAV.D_prop**4.0 * C_T
            torque_prop = MAV.rho * (omega / (2.0 * np.pi))**2.0 * MAV.D_prop**5.0 * C_Q
        else:
            thrust_prop = 0.0
            torque_prop = 0.0

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        # NED down velocity is negative during a positive-angle climb.
        self.true_state.gamma = np.arctan2(-pdot.item(2), np.linalg.norm(pdot[0:2]))
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.gimbal_az = 0
        self.true_state.gimbal_el = 0
