"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)

        # instantiate TECS controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        # throttle gains (per unit of specific energy error, J/kg = m^2/s^2)
        self.E_kp = 4.0e-3
        self.E_ki = 1.5e-3
        # pitch gains (rad per unit of energy distribution error)
        self.L_kp = 1.2e-3
        self.L_ki = 2.4e-4
        # saturated altitude error
        self.h_error_max = 20.0  # meters
        self.E_integrator = 0
        self.L_integrator = 0
        self.E_error_d1 = 0
        self.L_error_d1 = 0
        self.delta_t_d1 = 0
        self.theta_c_d1 = 0
        self.theta_c_max = np.radians(30)
        self.Ts = ts_control
        self.commanded_state = MsgState()

    def update(self, cmd, state):

        # lateral autopilot (identical to successive loop closure)
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = self.course_from_roll.update(chi_c, state.chi)
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal TECS autopilot (slides 48-54)
        #   specific total energy     E = g*h + Va^2/2
        #   energy distribution       L = g*h - Va^2/2
        # throttle balances the total energy, pitch balances the distribution
        h_error = self.saturate(cmd.altitude_command - state.altitude,
                                -self.h_error_max, self.h_error_max)
        h_reference = state.altitude + h_error
        E_c = MAV.gravity * h_reference + 0.5 * cmd.airspeed_command**2.0
        E = MAV.gravity * state.altitude + 0.5 * state.Va**2.0
        L_c = MAV.gravity * h_reference - 0.5 * cmd.airspeed_command**2.0
        L = MAV.gravity * state.altitude - 0.5 * state.Va**2.0

        E_error = E_c - E
        L_error = L_c - L

        # trapezoidal integrators
        self.E_integrator = self.E_integrator \
                            + (self.Ts/2.) * (E_error + self.E_error_d1)
        self.L_integrator = self.L_integrator \
                            + (self.Ts/2.) * (L_error + self.L_error_d1)

        # throttle from total energy error (trim feedforward)
        delta_t = self.E_kp * E_error \
                  + self.E_ki * self.E_integrator \
                  + AP.throttle_trim
        delta_t_sat = self.saturate(delta_t, 0.0, 1.0)
        # integral anti-windup (back-calculation)
        if np.abs(self.E_ki) > 1e-12:
            self.E_integrator = self.E_integrator \
                                + (self.Ts / self.E_ki) * (delta_t_sat - delta_t)

        # pitch command from energy distribution error (trim feedforward)
        theta_c = self.L_kp * L_error \
                  + self.L_ki * self.L_integrator \
                  + AP.theta_trim
        theta_c_sat = self.saturate(theta_c, -self.theta_c_max, self.theta_c_max)
        if np.abs(self.L_ki) > 1e-12:
            self.L_integrator = self.L_integrator \
                                + (self.Ts / self.L_ki) * (theta_c_sat - theta_c)

        delta_e = self.pitch_from_elevator.update(
                      theta_c_sat, state.theta, state.q) \
                  + AP.elevator_trim

        # update delayed variables
        self.E_error_d1 = E_error
        self.L_error_d1 = L_error
        self.delta_t_d1 = delta_t_sat
        self.theta_c_d1 = theta_c_sat

        # construct output and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t_sat)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c_sat
        self.commanded_state.chi = chi_c
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output
