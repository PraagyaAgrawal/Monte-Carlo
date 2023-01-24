#Simulating a game where the probability of winning each round is p% and the game ends when you lose 2 rounds in a row.
#The objective is to find the expected number of rounds until the game ends.

#Importing modules
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Creating a list to store the number of rounds until the game ended in each iteration
rounds = []

#Taking the input of p and the number of iterations wanted
p = int(input("Probability of winning a round (Percentage): "))
print("The game ends when you lose two rounds in a row", "\n")
n = int(input("Enter number of iterations: "))

#Simulating the game n times and storing the result after each iteration into rounds
for i in range(n):
	end_condition = 0
	round_count = 0
	while True:
		round_count += 1
		x = random.randint(1,100)
		if x>p:
			if end_condition == 1:
				rounds.append(round_count)
				break
			else:
				end_condition = 1
		else:
			end_condition = 0

#Plotting the results and displaying the mean
plt.style.use("dark_background")
plt.figure()
plt.xlabel("Number of rounds until game ended")
plt.ylabel("Frequency")
sns.distplot(rounds, bins = "sturges", kde=False)
plt.axvline(np.array(rounds).mean(), color = "red", label = f"The mean is {np.array(rounds).mean():.0f}")
plt.legend()
plt.show()