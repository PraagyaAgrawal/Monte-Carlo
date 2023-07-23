import matplotlib.pyplot as plt

# Function to plot the Ising model on a 1D lattice
def plot_1d_ising():
    n_sites = 10  # Number of sites in the 1D lattice
    positions = [(i, 0) for i in range(n_sites)]  # Coordinates of lattice sites

    plt.figure(figsize=(5, 1))
    for pos in positions:
        plt.plot(pos[0], pos[1], 'o', markersize=2, color='black', fillstyle='full')
    
    for i in range(n_sites - 1):
        x_coords = [positions[i][0], positions[i + 1][0]]
        y_coords = [positions[i][1], positions[i + 1][1]]
        plt.plot(x_coords, y_coords, linewidth=1, color='black')
    
    plt.axis('off')
    plt.show()

# Function to plot the Ising model on a 2D lattice
def plot_2d_ising():
    n_rows, n_cols = 6, 6  # Number of rows and columns in the 2D lattice
    positions = [(i, j) for i in range(n_rows) for j in range(n_cols)]  # Coordinates of lattice sites

    plt.figure(figsize=(5, 5))
    for pos in positions:
        plt.plot(pos[1], pos[0], 'o', markersize=2, color='black', fillstyle='full')
    
    for i in range(n_rows):
        for j in range(n_cols):
            if j < n_cols - 1:
                x_coords = [positions[i * n_cols + j][1], positions[i * n_cols + j + 1][1]]
                y_coords = [positions[i * n_cols + j][0], positions[i * n_cols + j + 1][0]]
                plt.plot(x_coords, y_coords, linewidth=1, color='black')
            
            if i < n_rows - 1:
                x_coords = [positions[i * n_cols + j][1], positions[(i + 1) * n_cols + j][1]]
                y_coords = [positions[i * n_cols + j][0], positions[(i + 1) * n_cols + j][0]]
                plt.plot(x_coords, y_coords, linewidth=1, color='black')
    
    plt.axis('off')
    plt.show()

# Function to plot the Ising model on a 3D lattice
def plot_3d_ising():
    n_layers, n_rows, n_cols = 3, 3, 3  # Number of layers, rows, and columns in the 3D lattice
    positions = [(i, j, k) for i in range(n_layers) for j in range(n_rows) for k in range(n_cols)]  # Coordinates of lattice sites

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for pos in positions:
        ax.scatter(pos[2], pos[1], pos[0], marker='o', s=20, c='black')
    
    for i in range(n_layers):
        for j in range(n_rows):
            for k in range(n_cols):
                if k < n_cols - 1:
                    x_coords = [positions[i * n_rows * n_cols + j * n_cols + k][2], positions[i * n_rows * n_cols + j * n_cols + k + 1][2]]
                    y_coords = [positions[i * n_rows * n_cols + j * n_cols + k][1], positions[i * n_rows * n_cols + j * n_cols + k + 1][1]]
                    z_coords = [positions[i * n_rows * n_cols + j * n_cols + k][0], positions[i * n_rows * n_cols + j * n_cols + k + 1][0]]
                    ax.plot(x_coords, y_coords, z_coords, linewidth=0.1, color='black')
                
                if j < n_rows - 1:
                    x_coords = [positions[i * n_rows * n_cols + j * n_cols + k][2], positions[i * n_rows * n_cols + (j + 1) * n_cols + k][2]]
                    y_coords = [positions[i * n_rows * n_cols + j * n_cols + k][1], positions[i * n_rows * n_cols + (j + 1) * n_cols + k][1]]
                    z_coords = [positions[i * n_rows * n_cols + j * n_cols + k][0], positions[i * n_rows * n_cols + (j + 1) * n_cols + k][0]]
                    ax.plot(x_coords, y_coords, z_coords, linewidth=0.1, color='black')
                
                if i < n_layers - 1:
                    x_coords = [positions[i * n_rows * n_cols + j * n_cols + k][2], positions[(i + 1) * n_rows * n_cols + j * n_cols + k][2]]
                    y_coords = [positions[i * n_rows * n_cols + j * n_cols + k][1], positions[(i + 1) * n_rows * n_cols + j * n_cols + k][1]]
                    z_coords = [positions[i * n_rows * n_cols + j * n_cols + k][0], positions[(i + 1) * n_rows * n_cols + j * n_cols + k][0]]
                    ax.plot(x_coords, y_coords, z_coords, linewidth=0.1, color='black')

    ax.axis('off')
    plt.show()

# Generate and display the Ising models for 1D, 2D, and 3D
plot_1d_ising()
plot_2d_ising()
plot_3d_ising()