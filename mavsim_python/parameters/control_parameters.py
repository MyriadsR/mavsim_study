import numpy as np
import models.model_coef as TF
import parameters.aerosonde_parameters as MAV


# trim feedforward values from the chapter 5 linearization
gravity = MAV.gravity  # gravity constant
Va0 = TF.Va_trim
rho = MAV.rho  # density of air
sigma = 0  # low pass filter gain for derivative
elevator_trim = TF.u_trim.item(0)
throttle_trim = TF.u_trim.item(3)
theta_trim = TF.theta_trim

#----------roll loop-------------
# get transfer function data for delta_a to phi:  phi/delta_a = a_phi2/(s(s+a_phi1))
# closed loop: s^2 + (a_phi1 + a_phi2*kd)s + a_phi2*kp = s^2 + 2*zeta*wn*s + wn^2
wn_roll = 8.0
zeta_roll = 0.707
roll_kp = wn_roll**2 / TF.a_phi2
roll_kd = (2*zeta_roll*wn_roll - TF.a_phi1) / TF.a_phi2

#----------course loop-------------
# with the roll loop closed, chi/phi ~ g/Va; place the course pole a factor
# of 5 below the roll bandwidth, PI to remove steady-state error
wn_course = wn_roll / 5.0
zeta_course = 0.707
course_kp = wn_course * Va0 / gravity
course_ki = course_kp * wn_course / 5.0

#----------yaw damper-------------
# washout filter on r (pole p_wo) with gain kr to add dutch-roll damping
yaw_damper_p_wo = 1.5
yaw_damper_kr = 0.15

#----------pitch loop-------------
# theta/delta_e = a_theta3/(s^2 + a_theta1*s + a_theta2),  a_theta3 < 0
# closed loop: s^2 + 2*zeta*wn*s + wn^2 with
#   pitch_kp = (wn^2 - a_theta2)/a_theta3,   pitch_kd = (2*zeta*wn - a_theta1)/a_theta3
# K_theta_DC precompensates the DC gain to unity
wn_pitch = 12.0
zeta_pitch = 0.707
pitch_kp = (wn_pitch**2 - TF.a_theta2) / TF.a_theta3
pitch_kd = (2*zeta_pitch*wn_pitch - TF.a_theta1) / TF.a_theta3
K_theta_DC = wn_pitch**2 / (wn_pitch**2 - TF.a_theta2)

#----------altitude loop-------------
# PI on altitude error -> commanded climb angle -> theta_c = gamma_c + theta_trim;
# the bandwidth must stay well below the pitch bandwidth for adequate damping
# of the phugoid-coupled altitude response (see linear closed-loop analysis)
wn_altitude = 1.0
zeta_altitude = 0.707
altitude_kp = wn_altitude / Va0
altitude_ki = altitude_kp * wn_altitude / 5.0
altitude_zone = 50.0  # altitude error beyond which the climb rate is saturated

#---------airspeed hold using throttle---------------
# Va/delta_t = a_V2/(s + a_V1); PI places both closed-loop poles
wn_airspeed_throttle = 2.0
zeta_airspeed_throttle = 1.0
airspeed_throttle_kp = (2*zeta_airspeed_throttle*wn_airspeed_throttle - TF.a_V1) / TF.a_V2
airspeed_throttle_ki = wn_airspeed_throttle**2 / TF.a_V2
