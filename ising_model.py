import numpy as np
#A 5x5 array, to simulate a 5x5 lattice
model = np.full((5,5), -1, dtype=int)

size = 4

#Components of hamiltonian: H = - sigma(E_ij) - sigma(J_i)
#J = energy associated with external field
def J_i(J, arr):
	sum1 = 0
	for i in arr:
		for j in i:
			sum1 += J*j
	return sum1

#E = energy associated with nearest neighbour interactions
def E_ij_horizontal(E, arr):
	sum2 = 0
	for i in arr:
		for j in range(size-1):
			sum2 += E*i[j]*i[j+1]
		sum2 += E*i[0]*i[-1]
	return sum2

def E_ij_vertical(E, arr):
	sum3 = 0
	for i in range(size-1):
		for j in range(size):
			sum3 += E*arr[i][j]*arr[i+1][j]
	for k in range(size):
		sum3 += E*arr[0][k]*arr[-1][k]
	return sum3

E = - 4 #eV
J = 0 #no external magnetic field

#Hamiltonian of the model
def H(E, J, arr):
	return - (E_ij_horizontal(E, arr) + E_ij_vertical(E, arr)) - J_i(J,arr)

#Changes the value of a lattice site from -1 to 1 and vice versa
def plus_one(x):
	if x == -1:
		x = 1
	else:
		x = -1
	return x

#Running this until complete = True will give all the possible values for the system, which will be used in the partition function Z
def change_model(arr):
	a = [size - 1, size - 1]
	while True:
		if arr[a[0]][a[1]] == 1:
			arr[a[0]][a[1]] = plus_one(arr[a[0]][a[1]])
			if a[1] != 0:
				a = [a[0], a[1] - 1]
			else:
				a = [a[0] - 1, size - 1]
		else:
			arr[a[0]][a[1]] = plus_one(arr[a[0]][a[1]])
			break

#Boltzmann constant
k = 1.380649 * 10**(-23)

#Partition function Z; T = temperature in Kelvin
def Z(T, E, J, arr):
	sum4 = 0
	for i in range(2**(size**2)):
		sum4 += np.exp(- H(E, J, arr) / (k*T) )
	return sum4

T = 293 #K, during STP

print(f"Z = {Z(T, E, J, model)}")