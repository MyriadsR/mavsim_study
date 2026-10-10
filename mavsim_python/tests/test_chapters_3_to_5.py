"""Numerical regression checks; run with unittest discovery, without a GUI."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from message_types.msg_delta import MsgDelta
from models.mav_dynamics import MavDynamics as RigidDynamics
from models.mav_dynamics_control import MavDynamics
from models.mav_dynamics_sensors import MavDynamics as SensorDynamics
from models.compute_models import compute_ss_model, compute_tf_model, df_dx, euler_state
from models.trim import compute_trim
from models.wind_simulation import WindSimulation
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_quaternion, quaternion_to_euler
from tools.transfer_function import TransferFunction


class ChapterRegressionTests(unittest.TestCase):
    def test_rigid_dynamics_match_inertia_matrix(self):
        rng = np.random.default_rng(23)
        inertia = np.array([[MAV.Jx, 0, -MAV.Jxz],
                            [0, MAV.Jy, 0], [-MAV.Jxz, 0, MAV.Jz]])
        mav = RigidDynamics(0.01)
        for _ in range(20):
            state = np.zeros((13, 1))
            state[3:6] = rng.normal(0, 20, (3, 1))
            state[6:10] = euler_to_quaternion(*rng.uniform(-1, 1, 3))
            state[10:13] = rng.normal(0, 0.5, (3, 1))
            forces = rng.normal(0, 10, (6, 1))
            velocity = state[3:6, 0]
            rates = state[10:13, 0]
            derivative = mav._f(state, forces)
            np.testing.assert_allclose(
                derivative[3:6, 0],
                forces[:3, 0] / MAV.mass - np.cross(rates, velocity), atol=1e-12)
            np.testing.assert_allclose(
                derivative[10:13, 0],
                np.linalg.solve(inertia, forces[3:, 0] - np.cross(rates, inertia @ rates)),
                atol=1e-12)

    def test_new_wind_affects_first_step(self):
        calm = MavDynamics(0.01)
        windy = MavDynamics(0.01)
        wind = np.zeros((6, 1))
        wind[0] = 10
        delta = MsgDelta(elevator=-0.124, throttle=0.77)
        calm.update(delta, np.zeros((6, 1)))
        windy.update(delta, wind)
        self.assertGreater(abs(windy._state.item(3) - calm._state.item(3)), 0.01)
        np.testing.assert_allclose(windy._wind, wind[:3], atol=1e-12)

    def test_flight_path_angle_climb_descent_and_rest(self):
        for model in (MavDynamics, SensorDynamics):
            mav = model(0.01)
            for degrees in (-10, 10):
                mav._state[6:10] = euler_to_quaternion(0, np.radians(degrees), 0)
                mav._update_velocity_data()
                mav._update_true_state()
                self.assertAlmostEqual(mav.true_state.gamma, np.radians(degrees))
            mav._state[3:6] = 0
            mav._update_true_state()
            self.assertEqual(mav.true_state.gamma, 0)

    def test_full_aircraft_rk4_converges_at_fourth_order(self):
        initial = MavDynamics(0.01)._state[:13].copy()
        initial[3:6] = [[25], [1], [2]]
        initial[6:10] = euler_to_quaternion(0.12, 0.06, -0.1)
        initial[10:13] = [[0.2], [0.15], [-0.1]]
        delta = MsgDelta(elevator=-0.139, aileron=0.005, rudder=0.003, throttle=0.76)
        wind = np.array([[2], [-1], [0], [0.1], [-0.1], [0.05]])
        reference_mav = MavDynamics(0.01)

        def rhs(time, state):
            reference_mav._state = state.reshape(13, 1)
            reference_mav._update_velocity_data(wind)
            return reference_mav._f(
                reference_mav._state, reference_mav._forces_moments(delta)).reshape(-1)

        solution = solve_ivp(rhs, (0, 1), initial.reshape(-1), rtol=1e-12, atol=1e-13)
        self.assertTrue(solution.success)
        errors = []
        for dt in (0.04, 0.02, 0.01):
            mav = MavDynamics(dt)
            mav._state[:13] = initial
            mav._state[13:15] = [[0.2], [-0.3]]
            for _ in range(round(1 / dt)):
                mav.update(delta, wind)
            errors.append(np.linalg.norm(mav._state[:13, 0] - solution.y[:, -1]))
            self.assertAlmostEqual(np.linalg.norm(mav._state[6:10]), 1)
            np.testing.assert_array_equal(mav._state[13:15], [[0.2], [-0.3]])
        for coarse, fine in zip(errors, errors[1:]):
            self.assertGreater(coarse / fine, 12)

    def test_trim_level_climb_and_descent(self):
        for degrees in (0, 5, -5):
            mav = MavDynamics(0.01)
            gamma = np.radians(degrees)
            with contextlib.redirect_stdout(io.StringIO()):
                state, delta = compute_trim(mav, 25, gamma)
            self.assertGreaterEqual(delta.throttle, 0)
            self.assertLessEqual(delta.throttle, 1)
            self.assertAlmostEqual(np.linalg.norm(state[3:6]), 25, places=6)
            self.assertAlmostEqual(np.linalg.norm(state[6:10]), 1, places=6)
            np.testing.assert_allclose(state[[4, 7, 9, 10, 11, 12]], 0, atol=1e-8)
            derivative = mav._f(state, mav._forces_moments(delta))
            derivative[2] += 25 * np.sin(gamma)
            self.assertLess(np.linalg.norm(derivative[2:]), 0.01)

    def test_infeasible_trim_rejected_and_state_restored(self):
        mav = MavDynamics(0.01)
        wind = np.array([[2], [0], [0], [0.1], [0], [0]])
        mav._update_velocity_data(wind)
        original_state = mav._state.copy()
        original_wind = mav._wind.copy()
        original_air_data = (mav._Va, mav._alpha, mav._beta)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "No usable trim"):
                compute_trim(mav, 25, np.radians(30))
        np.testing.assert_array_equal(mav._state, original_state)
        np.testing.assert_array_equal(mav._wind, original_wind)
        self.assertEqual((mav._Va, mav._alpha, mav._beta), original_air_data)

    def test_unsuccessful_optimizer_is_rejected(self):
        mav = MavDynamics(0.01)
        original = mav._state.copy()

        def failed_optimizer(*args, **kwargs):
            mav._state[:] = 42
            return SimpleNamespace(success=False, message="iteration limit")

        with patch("models.trim.minimize", side_effect=failed_optimizer):
            with self.assertRaisesRegex(RuntimeError, "iteration limit"):
                compute_trim(mav, 25, 0)
        np.testing.assert_array_equal(mav._state, original)

    def test_trim_rejects_invalid_requests(self):
        for Va, gamma in [(0, 0), (-1, 0), (np.nan, 0), (25, np.nan), (25, np.pi/2)]:
            with self.assertRaises(ValueError):
                compute_trim(MavDynamics(0.01), Va, gamma)

    def test_euler_jacobian_has_correct_zero_bank_kinematics(self):
        mav = MavDynamics(0.01)
        state = mav._state[:13].copy()
        pitch = 0.05
        state[6:10] = euler_to_quaternion(0, pitch, 0)
        jacobian = df_dx(mav, euler_state(state), MsgDelta())
        # At zero bank: phi_dot=p+r*tan(theta), theta_dot=q,
        # psi_dot=r/cos(theta), irrespective of the numerical reference file.
        np.testing.assert_allclose(jacobian[6:9, 9:12],
                                   [[1, 0, np.tan(pitch)], [0, 1, 0],
                                    [0, 0, 1 / np.cos(pitch)]], atol=1e-12)

    def test_trim_linear_models_match_dimensional_stability_derivatives(self):
        mav = MavDynamics(0.01)
        with contextlib.redirect_stdout(io.StringIO()):
            state, delta = compute_trim(mav, 25, 0)
        matrices = compute_ss_model(mav, state, delta)
        coefficients = compute_tf_model(mav, state, delta)
        for matrix in matrices:
            self.assertTrue(np.all(np.isfinite(matrix)))
        speed = coefficients[0]
        roll_damping = -MAV.rho * speed * MAV.S_wing * MAV.b**2 * MAV.C_p_p / 4
        roll_control = MAV.rho * speed**2 * MAV.S_wing * MAV.b * MAV.C_p_delta_a / 2
        pitch_damping = -MAV.rho * speed * MAV.S_wing * MAV.c**2 * MAV.C_m_q / (4 * MAV.Jy)
        pitch_stiffness = -MAV.rho * speed**2 * MAV.S_wing * MAV.c * MAV.C_m_alpha / (2 * MAV.Jy)
        pitch_control = MAV.rho * speed**2 * MAV.S_wing * MAV.c * MAV.C_m_delta_e / (2 * MAV.Jy)
        np.testing.assert_allclose(coefficients[3:8],
                                   [roll_damping, roll_control, pitch_damping,
                                    pitch_stiffness, pitch_control], atol=1e-10)

    def test_quaternion_conversion_handles_scaled_and_vertical_attitudes(self):
        angles = (0.2, np.radians(40), -0.3)
        np.testing.assert_allclose(
            quaternion_to_euler(1.3 * euler_to_quaternion(*angles)), angles, atol=1e-14)
        for pitch in (-np.pi/2, np.pi/2):
            self.assertAlmostEqual(quaternion_to_euler(euler_to_quaternion(0, pitch, 0))[1], pitch)
        with self.assertRaises(ValueError):
            quaternion_to_euler(np.zeros((4, 1)))

    def test_nonmonic_transfer_functions_match_frequency_response(self):
        for numerator, denominator in [([1], [2, 2]), ([2, 4], [2, 6]),
                                       ([2, 4], [2, 8, 10, 12])]:
            transfer = TransferFunction(np.array([numerator], float),
                                        np.array([denominator], float), 0.01)
            for omega in (0, 0.1, 1, 10):
                s = 1j * omega
                response = (transfer.C @ np.linalg.solve(
                    s * np.eye(len(transfer.A)) - transfer.A, transfer.B)).item() + transfer.D
                expected = np.polyval(numerator, s) / np.polyval(denominator, s)
                self.assertAlmostEqual(abs(response - expected), 0, places=12)

    def test_dryden_filters_match_chapter_four(self):
        wind = WindSimulation(0.01)
        for transfer, length, sigma, lateral in [
                (wind.u_w, 200, 1.06, False), (wind.v_w, 200, 1.06, True),
                (wind.w_w, 50, 0.7, True)]:
            for omega in (0.01, 0.1, 1, 10):
                s = 1j * omega
                response = (transfer.C @ np.linalg.solve(
                    s * np.eye(len(transfer.A)) - transfer.A, transfer.B)).item() + transfer.D
                gain = sigma * np.sqrt((3 if lateral else 2) * 25 / (np.pi * length))
                expected = (gain * (s + 25 / (np.sqrt(3) * length)) / (s + 25 / length)**2
                            if lateral else gain / (s + 25 / length))
                self.assertAlmostEqual(abs(response - expected), 0, places=12)


if __name__ == "__main__":
    unittest.main()
