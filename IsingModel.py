#Imports
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Creating NxN sized 2D lattices
def initialize_lattices(N):
  
	init_random = np.random.random((N,N)) #Used to initialize the lattices

	#Lattice with net positive spin
	lattice_p = np.zeros((N, N))
	lattice_p[init_random>=0.25] = 1
	lattice_p[init_random<0.25] = -1
  
	return lattice_p

N = 50 #50x50 lattice
lattice_p = initialize_lattices(N)

#Function to get the energy f a lattice
def get_energy(lattice):
	kern = generate_binary_structure(2, 1)
	kern[1][1] = False
	arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
	return arr.sum() / 2

#Function corresponding to the Metropolis Algorithm while removing spins from the entire lattice
@numba.njit #Improving time complexity
def metropolis_whole(spin_arr, times, B, E, energy, remove): #Function to compute equilibrium
	spin_arr = spin_arr.copy() #Making a copy
	net_spins = np.zeros(times-1) #Array to store change in net spin over time
	net_energy = np.zeros(times-1) #Array to store change in net energy over time
	for r in range(remove):
		val = 0
		while val == 0:
			x = np.random.randint(0,50)
			y = np.random.randint(0,50)
			val = spin_arr[x,y]
		spin_arr[x,y] = 0
	for t in range(0,times-1):
		#Pick random point on array and flip spin
		x = np.random.randint(0,N)
		y = np.random.randint(0,N)
		spin_i = spin_arr[x,y] #initial spin
		spin_f = spin_i*-1 #proposed spin flip

		#Change in energy
		E_i = 0
		E_f = 0
		if x>0: #Only if not on right corner
			E_i += -spin_i*spin_arr[x-1,y]
			E_f += -spin_f*spin_arr[x-1,y]
		if x<N-1: #Only if not on left corner
			E_i += -spin_i*spin_arr[x+1,y]
			E_f += -spin_f*spin_arr[x+1,y]
		if y>0: #Only if not on top corner
			E_i += -spin_i*spin_arr[x,y-1]
			E_f += -spin_f*spin_arr[x,y-1]
		if y<N-1: #Only if not on bottom corner
			E_i += -spin_i*spin_arr[x,y+1]
			E_f += -spin_f*spin_arr[x,y+1]
        
		#Change state
		dE = E*(E_f-E_i)
		if (dE>0)*(np.random.random() < np.exp(-B*dE)):
			spin_arr[x,y]=spin_f
			energy += dE
		elif dE<=0:
			spin_arr[x,y]=spin_f
			energy += dE
            
		net_spins[t] = spin_arr.sum() #Adding new net spin to the array
		net_energy[t] = energy #Adding new new energy to the array
            
	return net_spins, net_energy

#Function corresponding to the Metropolis Algorithm while removing spins from the centre for the lattice
@numba.njit #Improving time complexity
def metropolis_centre(spin_arr, times, B, E, energy, remove): #Function to compute equilibrium
	spin_arr = spin_arr.copy() #Making a copy
	net_spins = np.zeros(times-1) #Array to store change in net spin over time
	net_energy = np.zeros(times-1) #Array to store change in net energy over time
	for r in range(remove):
		val = 0
		while val == 0:
			x = np.random.randint(15,35)
			y = np.random.randint(`5,35)
			val = spin_arr[x,y]
		spin_arr[x,y] = 0
	for t in range(0,times-1):
		#Pick random point on array and flip spin
		x = np.random.randint(0,N)
		y = np.random.randint(0,N)
		spin_i = spin_arr[x,y] #initial spin
		spin_f = spin_i*-1 #proposed spin flip

		#Change in energy
		E_i = 0
		E_f = 0
		if x>0: #Only if not on right corner
			E_i += -spin_i*spin_arr[x-1,y]
			E_f += -spin_f*spin_arr[x-1,y]
		if x<N-1: #Only if not on left corner
			E_i += -spin_i*spin_arr[x+1,y]
			E_f += -spin_f*spin_arr[x+1,y]
		if y>0: #Only if not on top corner
			E_i += -spin_i*spin_arr[x,y-1]
			E_f += -spin_f*spin_arr[x,y-1]
		if y<N-1: #Only if not on bottom corner
			E_i += -spin_i*spin_arr[x,y+1]
			E_f += -spin_f*spin_arr[x,y+1]
        
		#Change state
		dE = E*(E_f-E_i)
		if (dE>0)*(np.random.random() < np.exp(-B*dE)):
			spin_arr[x,y]=spin_f
			energy += dE
		elif dE<=0:
			spin_arr[x,y]=spin_f
			energy += dE
            
		net_spins[t] = spin_arr.sum() #Adding new net spin to the array
		net_energy[t] = energy #Adding new new energy to the array
            
	return net_spins, net_energy

#Function corresponding to the Metropolis Algorithm while removing spins from outside the centre for the lattice
@numba.njit #Improving time complexity
def metropolis_edge(spin_arr, times, B, E, energy, remove): #Function to compute equilibrium
	spin_arr = spin_arr.copy() #Making a copy
	net_spins = np.zeros(times-1) #Array to store change in net spin over time
	net_energy = np.zeros(times-1) #Array to store change in net energy over time
	for r in range(remove):
		val = 0
		while val == 0:
			rand = np.random.randint(1,4)
			if rand == 1:
				x = np.random.randint(0,35)
				y = np.random.randint(0,15)
			elif rand == 2:
				x = np.random.randint(35,50)
				y = np.random.randint(0,35)     
			elif rand == 3:
				x = np.random.randint(15,50)
				y = np.random.randint(35,50)
			else:
				x = np.random.randint(0,15)
				y = np.random.randint(15,50)
			val = spin_arr[x,y]
		spin_arr[x,y] = 0
	for t in range(0,times-1):
		#Pick random point on array and flip spin
		x = np.random.randint(0,N)
		y = np.random.randint(0,N)
		spin_i = spin_arr[x,y] #initial spin
		spin_f = spin_i*-1 #proposed spin flip

		#Change in energy
		E_i = 0
		E_f = 0
		if x>0: #Only if not on right corner
			E_i += -spin_i*spin_arr[x-1,y]
			E_f += -spin_f*spin_arr[x-1,y]
		if x<N-1: #Only if not on left corner
			E_i += -spin_i*spin_arr[x+1,y]
			E_f += -spin_f*spin_arr[x+1,y]
		if y>0: #Only if not on top corner
			E_i += -spin_i*spin_arr[x,y-1]
			E_f += -spin_f*spin_arr[x,y-1]
		if y<N-1: #Only if not on bottom corner
			E_i += -spin_i*spin_arr[x,y+1]
			E_f += -spin_f*spin_arr[x,y+1]
        
		#Change state
		dE = E*(E_f-E_i)
		if (dE>0)*(np.random.random() < np.exp(-B*dE)):
			spin_arr[x,y]=spin_f
			energy += dE
		elif dE<=0:
			spin_arr[x,y]=spin_f
			energy += dE
            
		net_spins[t] = spin_arr.sum() #Adding new net spin to the array
		net_energy[t] = energy #Adding new new energy to the array
            
	return net_spins, net_energy

E = 4 #Energy associated with nearest-neighbor interactions
B = 0.05 #B = 1/kT

#Functions to display results for removal from outisde the centre
def plot_edge():
	spin_Rs_avg = np.zeros(41)
	energy_Rs_avg = np.zeros(41)
	for i in range(10):
		Rs = np.arange(0,401,10)
		spin_Rs = []
		energy_Rs = []
		for R in Rs:
			spins, energies = metropolis_edge(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
			spin_Rs.append(spins[-100000:].mean())
			energy_Rs.append(energies[-100000:].mean())
		spin_Rs = np.array(spin_Rs)/N**2
		energy_Rs = np.array(energy_Rs)
		spin_Rs_avg += spin_Rs
		energy_Rs_avg += energy_Rs
	spin_Rs_avg /= 10
	energy_Rs_avg /= 10
	plt.figure(figsize=(20,5))
	plt.title("Evolution of spin with removal from outside the centre of lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Average spin")
	plt.plot(Rs, spin_Rs_avg)
	plt.show()
	Original_spin = spin_Rs_avg[0]
	Max_spin = spin_Rs_avg.max()
	Min_spin = spin_Rs_avg.min()
	print(f"Original= {Original_spin}, Max = {Max_spin}, Min = {Min_spin}")
	plt.figure(figsize=(20,5))
	plt.title("Evolution of energy with removal from outside the centre of lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Net energy")
	plt.plot(Rs, energy_Rs_avg)
	plt.show()
	Original_energy = energy_Rs_avg[0]
	Max_energy = energy_Rs_avg.max()
	Min_energy = energy_Rs_avg.min()
	print(f"Original = {Original_energy}, Max = {Max_energy}, Min = {Min_energy}")

#Functions to display results for removal from the centre
def plot_centre():
	spin_Rs_avg = np.zeros(41)
	energy_Rs_avg = np.zeros(41)
	for i in range(10):
		Rs = np.arange(0,401,10)
		spin_Rs = []
		energy_Rs = []
		for R in Rs:
			spins, energies = metropolis_centre(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
			spin_Rs.append(spins[-100000:].mean())
			energy_Rs.append(energies[-100000:].mean())
		spin_Rs = np.array(spin_Rs)/N**2
		energy_Rs = np.array(energy_Rs)
		spin_Rs_avg += spin_Rs
		energy_Rs_avg += energy_Rs
	spin_Rs_avg /= 10
	energy_Rs_avg /= 10
	plt.figure(figsize=(20,5))
	plt.title("Evolution of spin with removal from the centre of lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Average spin")
	plt.plot(Rs, spin_Rs_avg)
	plt.show()
	Original_spin = spin_Rs_avg[0]
	Max_spin = spin_Rs_avg.max()
	Min_spin = spin_Rs_avg.min()
	print(f"Original= {Original_spin}, Max = {Max_spin}, Min = {Min_spin}")
	plt.figure(figsize=(20,5))
	plt.title("Evolution of energy with removal from the centre of lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Net energy")
	plt.plot(Rs, energy_Rs_avg)
	plt.show()
	Original_energy = energy_Rs_avg[0]
	Max_energy = energy_Rs_avg.max()
	Min_energy = energy_Rs_avg.min()
	print(f"Original = {Original_energy}, Max = {Max_energy}, Min = {Min_energy}")

#Functions to display results for removal from the whole lattice
def plot_whole():
	spin_Rs_avg = np.zeros(41)
	energy_Rs_avg = np.zeros(41)
	for i in range(10):
		Rs = np.arange(0,2001,50)
		spin_Rs = []
		energy_Rs = []
		for R in Rs:
			spins, energies = metropolis_whole(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
			spin_Rs.append(spins[-100000:].mean())
			energy_Rs.append(energies[-100000:].mean())
		spin_Rs = np.array(spin_Rs)/N**2
		energy_Rs = np.array(energy_Rs)
		spin_Rs_avg += spin_Rs
		energy_Rs_avg += energy_Rs
	spin_Rs_avg /= 10
	energy_Rs_avg /= 10
	plt.figure(figsize=(20,5))
	plt.title("Evolution of spin with removal from the lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Average spin")
	plt.plot(Rs, spin_Rs_avg)
	plt.show()
	Original_spin = spin_Rs_avg[0]
	Max_spin = spin_Rs_avg.max()
	Min_spin = spin_Rs_avg.min()
	print(f"Original= {Original_spin}, Max = {Max_spin}, Min = {Min_spin}")
	plt.figure(figsize=(20,5))
	plt.title("Evolution of energy with removal from lattice_p")
	plt.xlabel("Number of sites removed")
	plt.ylabel("Net energy")
	plt.plot(Rs, energy_Rs_avg)
	plt.show()
	Original_energy = energy_Rs_avg[0]
	Max_energy = energy_Rs_avg.max()
	Min_energy = energy_Rs_avg.min()
	print(f"Original = {Original_energy}, Max = {Max_energy}, Min = {Min_energy}")

#Testing model accurary while removing from whole lattice
def model_whole():
  spin_Rs_avg = np.zeros(41)
  energy_Rs_avg = np.zeros(41)
  for i in range(10):
    Rs = np.arange(0,2001,50)
    spin_Rs = []
    energy_Rs = []
    for R in Rs:
      spins, energies = metropolis_whole(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
      spin_Rs.append((np.array(spins[-100000:])/N**2).mean())
      energy_Rs.append((np.array(energies[-100000:])/N**2).mean())
    spin_Rs_avg += spin_Rs
    energy_Rs_avg += energy_Rs
  spin_Rs_avg /= 10
  energy_Rs_avg /= 10
  spin_Rs_1 = spin_Rs_avg[:-1]
  spin_Rs_2 = spin_Rs_avg[1:]
  spin_x_train, spin_x_test, spin_y_train, spin_y_test = train_test_split(spin_Rs_1, spin_Rs_2, test_size = 0.33, random_state = 42)
  model_spin = LinearRegression().fit(spin_x_train.reshape(-1,1), spin_y_train)
  print(model_spin.score(spin_x_test.reshape(-1,1), spin_y_test))
  print(model_spin.coef_)
  print(model_spin.intercept_)
  energy_Rs_1 = energy_Rs_avg[:-1]
  energy_Rs_2 = energy_Rs_avg[1:]
  energy_x_train, energy_x_test, energy_y_train, energy_y_test = train_test_split(energy_Rs_1, energy_Rs_2, test_size = 0.33, random_state = 42)
  model_energy = LinearRegression().fit(energy_x_train.reshape(-1,1), energy_y_train)
  print(model_energy.score(energy_x_test.reshape(-1,1), energy_y_test))
  print(model_energy.coef_)
  print(model_energy.intercept_)

#Testing model accurary while removing from the centre of the lattice
def model_centre():
  spin_Rs_avg = np.zeros(41)
  energy_Rs_avg = np.zeros(41)
  for i in range(10):
    Rs = np.arange(0,401,10)
    spin_Rs = []
    energy_Rs = []
    for R in Rs:
      spins, energies = metropolis_centre(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
      spin_Rs.append((np.array(spins[-100000:])/N**2).mean())
      energy_Rs.append((np.array(energies[-100000:])/N**2).mean())
    spin_Rs_avg += spin_Rs
    energy_Rs_avg += energy_Rs
  spin_Rs_avg /= 10
  energy_Rs_avg /= 10
  spin_Rs_1 = spin_Rs_avg[:-1]
  spin_Rs_2 = spin_Rs_avg[1:]
  spin_x_train, spin_x_test, spin_y_train, spin_y_test = train_test_split(spin_Rs_1, spin_Rs_2, test_size = 0.33, random_state = 42)
  model_spin = LinearRegression().fit(spin_x_train.reshape(-1,1), spin_y_train)
  print(model_spin.score(spin_x_test.reshape(-1,1), spin_y_test))
  print(model_spin.coef_)
  print(model_spin.intercept_)
  energy_Rs_1 = energy_Rs_avg[:-1]
  energy_Rs_2 = energy_Rs_avg[1:]
  energy_x_train, energy_x_test, energy_y_train, energy_y_test = train_test_split(energy_Rs_1, energy_Rs_2, test_size = 0.33, random_state = 42)
  model_energy = LinearRegression().fit(energy_x_train.reshape(-1,1), energy_y_train)
  print(model_energy.score(energy_x_test.reshape(-1,1), energy_y_test))
  print(model_energy.coef_)
  print(model_energy.intercept_)

#Testing model accurary while removing from outside the centre of the lattice
def model_edge():
  spin_Rs_avg = np.zeros(41)
  energy_Rs_avg = np.zeros(41)
  for i in range(10):
    Rs = np.arange(0,401,10)
    spin_Rs = []
    energy_Rs = []
    for R in Rs:
      spins, energies = metropolis_edge(lattice_p, 1000000, B, E, get_energy(lattice_p), R)
      spin_Rs.append((np.array(spins[-100000:])/N**2).mean())
      energy_Rs.append((np.array(energies[-100000:])/N**2).mean())
    spin_Rs_avg += spin_Rs
    energy_Rs_avg += energy_Rs
  spin_Rs_avg /= 10
  energy_Rs_avg /= 10
  spin_Rs_1 = spin_Rs_avg[:-1]
  spin_Rs_2 = spin_Rs_avg[1:]
  spin_x_train, spin_x_test, spin_y_train, spin_y_test = train_test_split(spin_Rs_1, spin_Rs_2, test_size = 0.33, random_state = 42)
  model_spin = LinearRegression().fit(spin_x_train.reshape(-1,1), spin_y_train)
  print(model_spin.score(spin_x_test.reshape(-1,1), spin_y_test))
  print(model_spin.coef_)
  print(model_spin.intercept_)
  energy_Rs_1 = energy_Rs_avg[:-1]
  energy_Rs_2 = energy_Rs_avg[1:]
  energy_x_train, energy_x_test, energy_y_train, energy_y_test = train_test_split(energy_Rs_1, energy_Rs_2, test_size = 0.33, random_state = 42)
  model_energy = LinearRegression().fit(energy_x_train.reshape(-1,1), energy_y_train)
  print(model_energy.score(energy_x_test.reshape(-1,1), energy_y_test))
  print(model_energy.coef_)
