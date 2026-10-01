"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag = True, steady_state = np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state

        #   Dryden gust model parameters (slide 40, Table 1:
        #   low altitude, light turbulence)
        Va = 25.0        # nominal airspeed used in the gust filters (m/s)
        Lu = 200.0       # longitudinal turbulence length scale (m)
        Lv = 200.0       # lateral turbulence length scale (m)
        Lw = 50.0        # vertical turbulence length scale (m)
        if gust_flag == True:
            sigma_u = 1.06   # m/s
            sigma_v = 1.06   # m/s
            sigma_w = 0.70   # m/s
        else:
            sigma_u = 0.0
            sigma_v = 0.0
            sigma_w = 0.0

        # Dryden gust transfer functions (slide 40), unit-variance white noise
        # in, gust velocities (body frame) out:
        #   Hu(s) = sigma_u sqrt(2 Va/(pi Lu)) / (s + Va/Lu)
        #   Hv(s) = sigma_v sqrt(3 Va/(pi Lv)) (s + Va/(sqrt(3) Lv)) / (s + Va/Lv)^2
        #   Hw(s) = sigma_w sqrt(3 Va/(pi Lw)) (s + Va/(sqrt(3) Lw)) / (s + Va/Lw)^2
        # the second-order numerators carry 2 coefficients (m < n) so the
        # control-canonical form has no direct feed-through of the noise input
        Ku = sigma_u * np.sqrt(2.0 * Va / (np.pi * Lu))
        num_u = np.array([[0.0, Ku]])
        den_u = np.array([[1.0, Va / Lu]])
        self.u_w = TransferFunction(num=num_u, den=den_u, Ts=Ts)

        def dryden_lateral_vertical(L, sigma):
            A = sigma * np.sqrt(3.0 * Va / (np.pi * L))
            num = np.array([[A, A * Va / (np.sqrt(3.0) * L)]])
            den = np.array([[1.0, 2.0 * Va / L, (Va / L)**2.0]])
            return TransferFunction(num=num, den=den, Ts=Ts)

        self.v_w = dryden_lateral_vertical(Lv, sigma_v)
        self.w_w = dryden_lateral_vertical(Lw, sigma_w)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

