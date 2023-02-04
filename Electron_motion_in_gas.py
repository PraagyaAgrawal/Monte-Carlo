#Imports
import numpy as np
import random
from math import log

iters = 1000000

#Length of electron free path
def L(T, P, d):
	return (4 * 1.38 * (10 ** (-28)) * T) / (np.pi * P * (d**2))

#Taking into account energy
def L_E(T, P, d, QE, Q_E):
	return L(T, P, d) * (Q_E / QE)

def lamda(T, P, d, QE, Q_E):
	return L_E(T, P, d, QE, Q_E) * log(random.uniform(0,1))

#Probability density function
def pdf(T, P, d, QE, Q_E):
	return (1 / L_E(T, P, d, QE, Q_E)) * np.exp( - lamda(T, P, d, QE, Q_E) / L_E(T, P, d, QE, Q_E) )

#Probability of elastic collision
def P_e(QeE, QE):
	return QeE/QE

#Probability of inelastic collision
def P_i(QeE, QE):
	return 1 - P_e(QeE, QE)

#Standard temperature and pressure
T = 293.15
P = 101325

#Diameter of nitrogen molecule
d = 3 * (10**(-10))

QE = 10 * (10**(-22)) #At E = 1 eV
Q_E = 12 * (10**(-22)) #At E = 3eV

L_total = 0

for i in range(iters):
	L_total += lamda(T, P, d, QE, Q_E)

print(f"The length of the electron free path in Nitrogen at STP and at energy of 1 eV is {(L_total / iters):.15f} m")