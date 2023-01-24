#Simualting a gas particle moving with drift movement due to a net force acting on it, and a random movement.
#The objective is to find the expected location of the particle relative to its original location after a given amount of time.

#Importing modules
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Configuring the graph where the results will be plotted
plt.style.use("dark_background")
plt.figure()
ax = plt.axes(projection ='3d')
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("z")


#Initialising the drift movement
drift_x = 7
drift_y = 5
drift_z = 4

#Initialising the number of iterations and the time after which the location is to be found
iters = 1000
total_time = 100

#Initialising the running total of the final position after every iteration, which will later be divided by the number of iterations to find the average
total_x = 0
total_y = 0
total_z = 0

for i in range(iters):
	#Initialising the lists that will store the changes in the particle's position after every unit of time
	position_x_lst = [0]
	position_y_lst = [0]
	position_z_lst = [0]

	for j in range(total_time):
		#Initialising the random movement of the particle
		random_x = random.randint(-5,5)
		random_y = random.randint(-5,5)
		random_z= random.randint(-5,5)

		#Calculating the net change in the particle's position
		change_x = drift_x + random_x
		change_y = drift_y + random_y
		change_z = drift_z + random_z

		#Adding the new position to the lists
		position_x_lst.append(position_x_lst[j] + change_x)
		position_y_lst.append(position_y_lst[j] + change_y)
		position_z_lst.append(position_z_lst[j] + change_z)

	#Adding the final position of the particle to the running total
	total_x += position_x_lst[-1]
	total_y += position_y_lst[-1]
	total_z += position_z_lst[-1]

	#Creating the plot to visualize the particle's movement
	ax.plot3D(position_x_lst, position_y_lst, position_z_lst)

#Finding the average final position of the particle
average_x = total_x/iters
average_y = total_y/iters
average_z = total_z/iters

#Displaying the graphs and the expected final position of the particle
plt.title(f"Average position after {total_time} units of time is ({average_x:.1f},{average_y:.1f},{average_z:.1f})")
plt.show()