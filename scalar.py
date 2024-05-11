# Re-importing necessary libraries and redefining the RungeKutta class and MSEIRS model parameters and equations
import numpy as np
import matplotlib.pyplot as plt
from runge import RungeKutta

# Implementing the SIRS model using the provided RungeKutta class.

# SIRS model parameters
beta_sirs = 0.001  # Infection rate
gamma_sirs = 5  # Recovery rate
xi_sirs = 120  # Rate of losing immunity and becoming susceptible again

# Initial population values for each compartment
initial_S_sirs = 9999
initial_I_sirs = 1
initial_R_sirs = 0
initial_values_sirs = np.array([initial_S_sirs, initial_I_sirs, initial_R_sirs])

# SIRS model differential equations

# Parameters for the periodic variation of the infection rate
beta_max = 3  # Maximum infection rate
beta_min = 0  # Minimum infection rate
period = 60  # Period of oscillation (e.g., 365 days for a yearly cycle)


# Function to calculate the varying infection rate
def beta_sirs_varying(t):
    # Oscillating between beta_min and beta_max in a sinusoidal pattern
    return beta_min + (beta_max - beta_min) * (1 + np.sin(2 * np.pi * t / period)) / 2


def sirs_model(t, y):
    b = beta_sirs_varying(t)
    S, I, R = y
    dSdt = -b * S * I / (S + I + R) + R / xi_sirs
    dIdt = b * S * I / (S + I + R) - I / gamma_sirs
    dRdt = I / gamma_sirs - R / xi_sirs
    return np.array([dSdt, dIdt, dRdt])


# Time span and RungeKutta instance for SIRS

steps = 10000
start_time = 0
end_time = 365

rk4_sirs = RungeKutta(steps, start_time, end_time, initial_values_sirs, sirs_model)

# Calculate the SIRS model
rk4_sirs.calculate()

# Extracting the results for plotting SIRS model
times_sirs = rk4_sirs.result[1]
values_sirs = np.array(rk4_sirs.result[2])

# Plotting the results of SIRS model

