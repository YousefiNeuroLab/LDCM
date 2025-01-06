# Class to store all state parameters
class State:
    def __init__(self, dim_state, dim_obs, mode, R, R0, n_steps):
        """
        Initialize the State object with simulation parameters.

        Parameters:
        dim_state (int): Dimension of the state space.
        dim_obs (int): Dimension of the observation space.
        mode (int): Simulation mode (1: Random walk, 2: Generalized).
        R (float or array): Observation noise covariance.
        R0 (float or array): Initial state covariance.
        n_steps (int): Number of time steps in the simulation.
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.mode = mode

        # Initialize simulation parameters
        self.x_init, self.mean, self.chol_cov_matrix, self.A, self.B, self.C, self.D, self.eigenvalue, \
        self.eigenvectors, self.cov_matrix, self.w, self.Q = self.create_simulation_params()

        self.R = R
        self.R0 = R0
        self.n_steps = n_steps

    def create_simulation_params(self):
        """
        Generate and initialize simulation parameters such as covariance matrices,
        eigenvalues, eigenvectors, and transition matrices.

        Returns:
        Tuple containing initialized parameters for simulation.
        """
        x_init = np.random.normal(loc=0, scale=2, size=self.dim_state)  # Initial value of x
        mean = np.zeros(self.dim_obs)  # Mean vector

        # Generate covariance matrix with low correlations
        cov_matrix = generate_low_correlation_cov_matrix(self.dim_obs, (1, 2), (-0.1, 0.1))
        Q = cov_matrix[0] if self.dim_obs == 1 else cov_matrix

        # Cholesky decomposition and eigenvalue decomposition
        chol_cov_matrix = np.linalg.cholesky(cov_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        eigenvalue = np.diag(eigenvalues)
        Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(eigenvalue)))
        w = np.dot(eigenvectors, Lambda_inv_sqrt)

        # Initialize observation matrices
        C = np.array([[2]])
        D = np.array([[-1]])

        # Initialize transition matrices based on mode
        if self.mode == 1:  # Random walk mode
            A = np.eye(self.dim_state)
            B = np.zeros(self.dim_state)
        elif self.mode == 2:  # Generalized mode
            A = np.random.normal(loc=0, scale=0.1, size=(self.dim_state, self.dim_state))
            B = np.random.normal(loc=0, scale=0.1, size=self.dim_state)

        return x_init, mean, chol_cov_matrix, A, B, C, D, eigenvalue, eigenvectors, cov_matrix, w, Q

# Main simulation function
def run_simulation(dim_state, dim_obs, mode, R, R0, n_steps, num_trials):
    """
    Run the simulation for a specified number of trials and return generated data.

    Parameters:
    dim_state (int): Dimension of the state space.
    dim_obs (int): Dimension of the observation space.
    mode (int): Simulation mode (1: Random walk, 2: Generalized).
    R (float or array): Observation noise covariance.
    R0 (float or array): Initial state covariance.
    n_steps (int): Number of time steps in each trial.
    num_trials (int): Number of trials to simulate.

    Returns:
    Tuple containing state and observation data for all trials, labels, and the state object.
    """
    state = State(dim_state, dim_obs, mode, R, R0, n_steps)
    X_all_trials, Y_all_trials, labels = [], [], []
    label_occurrences = {0: 0, 1: 0}
    target_count = num_trials // 2

    while not all(count >= target_count for count in label_occurrences.values()):
        x, y = generate_simulation_data(state, n_steps - 1)
        y = np.squeeze(y, axis=-1)
        label = process_trials(np.array(y), num_trials, n_steps)

        if label_occurrences[label] < target_count:
            if label == 0:
                y += 20

            X_all_trials.append(x)
            Y_all_trials.append(y)
            labels.append(label)
            label_occurrences[label] += 1

    X_all_trials, Y_all_trials, labels = np.array(X_all_trials), np.array(Y_all_trials), np.array(labels)

    visualize_simulation(Y_all_trials[0], dim_obs)
    visualize_simulation(Y_all_trials[1], dim_obs)
    visualize_x(X_all_trials[0], dim_state)

    return X_all_trials, Y_all_trials, labels, state

# Function to generate simulation data
def generate_simulation_data(state: State, n_steps=99):
    """
    Generate simulation data for a single trial.

    Parameters:
    state (State): The state object containing simulation parameters.
    n_steps (int): Number of time steps to simulate.

    Returns:
    Tuple containing state and observation data for the trial.
    """
    x, y = [], []
    e = mvnrand(0, state.R, state.dim_state)
    v = mvnrand(0, state.Q, state.dim_obs)

    x_temp = state.A @ state.x_init.T + state.B + e
    x.append(x_temp)
    y_temp = state.C @ state.x_init + state.D + v[0]
    y.append(np.array(y_temp))

    for i in range(n_steps):
        e = mvnrand(0, state.R, state.dim_state)
        v = mvnrand(0, state.Q, state.dim_obs)
        x_temp = state.A @ x[i].T + state.B + e
        x.append(x_temp)
        y_temp = state.C @ x[i] + state.D + v[0]
        y.append(np.array(y_temp))

    return x, y

# Module: visualize
def visualize_simulation(y, n):
    """
    Visualize the elements of the observation vector over time.

    Parameters:
    y (list): Observation data for the trial.
    n (int): Number of observation dimensions to visualize.

    Returns:
    None
    """
    for i in range(n):
        element_data = [item[i] for item in y]
        plt.figure(figsize=(10, 6))
        plt.plot(element_data, marker='o', linestyle='-', label=f'Element {i+1}')
        plt.title(f"Observation Dimension {i+1} Over Time")
        plt.xlabel("Iteration")
        plt.ylabel(f"Dimension {i+1} Value")
        plt.grid(True)
        plt.legend()
        plt.show()

# Module: visualize_x
def visualize_x(x, n):
    """
    Visualize the state dimensions over time.

    Parameters:
    x (list): State data for the trial.
    n (int): Number of state dimensions to visualize.

    Returns:
    None
    """
    for i in range(n):
        plt.figure(figsize=(10, 6))
        plt.plot([state[i] for state in x], marker='o', linestyle='-', label=f'State {i+1}', color=f'C{i}')
        plt.title(f"State Dimension {i+1} Over Time")
        plt.xlabel("Iteration")
        plt.ylabel(f"State {i+1} Value")
        plt.grid(True)
        plt.legend()
        plt.show()

