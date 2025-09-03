import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import os
from mpl_toolkits.mplot3d import Axes3D

class JacobianComputer:
    """
    Computes the Jacobian matrix of the model's output with respect to its parameters using autograd.
    This implementation does not use functorch and instead computes gradients manually for better
    control over stability.
    """
    def __init__(self, model: nn.Module):
        self.model = model  # The neural network model for which we compute the Jacobian

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        # Put the model in training mode to ensure gradients are tracked.
        self.model.train()
        # Clear any existing gradients in the model.
        self.model.zero_grad()

        # Clone the input, detach it from any previous computation, and set requires_grad=True.
        x_input = x.detach().clone().requires_grad_(True)
        # Perform a forward pass with an added batch dimension, then remove it with squeeze.
        output = self.model(x_input.unsqueeze(0)).squeeze()

        # Determine the number of scalar outputs and total parameters in the model.
        n_outputs = output.numel()
        n_params = sum(p.numel() for p in self.model.parameters())
        # Initialize the Jacobian matrix with zeros.
        jacobian = torch.zeros(n_outputs, n_params, device=x.device, dtype=x.dtype)

        # Loop over each output element to compute its gradient with respect to the parameters.
        for i in range(n_outputs):
            # Clear gradients from previous iterations.
            self.model.zero_grad()
            # Create a fresh input for proper gradient tracking.
            x_fresh = x.detach().clone().requires_grad_(True)
            output_fresh = self.model(x_fresh.unsqueeze(0)).squeeze()
            
            # Select the i-th output element.
            if n_outputs > 1:
                output_scalar = output_fresh.view(-1)[i]
            else:
                output_scalar = output_fresh
            
            try:
                # Backpropagate to compute gradients of the selected output.
                output_scalar.backward(retain_graph=False)
                # Gather gradients from all parameters, flattening them into a single vector.
                param_grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_grads.append(param.grad.view(-1).clone())
                    else:
                        # If no gradient is computed for the parameter, use a zero tensor.
                        param_grads.append(torch.zeros_like(param).view(-1))
                # Concatenate gradients and assign to the corresponding row in the Jacobian.
                if param_grads:
                    jacobian[i, :] = torch.cat(param_grads)
            except RuntimeError:
                # In case of an error during gradient computation, set the row to zeros.
                jacobian[i, :] = 0.0

        # Clear gradients for safety and return the Jacobian detached from the graph.
        self.model.zero_grad()
        return jacobian.detach()

class ThermodynamicNeuralNetwork(nn.Module):
    """
    PyTorch implementation of the thermodynamic neural network for the 5-state control system.
    Uses the same architecture as the original NumPy implementation but with PyTorch tensors.
    """
    def __init__(self, num_inputs=5, num_outputs=5, num_layers=9, num_neurons=10):
        super(ThermodynamicNeuralNetwork, self).__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(num_inputs, num_neurons))
        layers.append(nn.Tanh())  # Using tanh as in original
        
        # Hidden layers  
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        
        # Output layer (no activation)
        layers.append(nn.Linear(num_neurons, num_outputs))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Kaiming-He initialization (similar to original)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming-He initialization for stability"""
        torch.manual_seed(0)  # For reproducibility
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming-He initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class PyTorchThermoDynamics:
    """PyTorch implementation of the system dynamics"""
    
    @staticmethod
    def state_dynamics_torch(positions, control_input=None, time=0.0):
        """
        Compute the drift vector field using PyTorch tensors
        positions: torch.Tensor of shape (5,)
        Returns: (dynamics, f) as torch.Tensors
        """
        if control_input is None:
            control_input = torch.zeros_like(positions)
            
        x1, x2, x3, x4, x5 = positions
        
        # Compute the drift vector field (same as NumPy version)
        f1 = 5 * np.tanh(50 * x1) * x5**2 + np.cos(x4)

        # f2: Combines a high-frequency cosine with a chaotic-like product of sines.
        f2 = np.cos(20 * x3) + 2 * np.sin(x1 * x2) * np.sin(x4 * x5)
            
        # f3: Introduces a localized feature using an exponential function.
        f3 = 10 * np.exp(-25 * x4**2) * x3 - 0.1 * x3**3

        # f4: A very high-frequency, complex coupling of four states.
        f4 = 2 * np.sin(15 * (x1 * x5 - x2 * x3))

        # f5: A steep tanh function acts like a switch.
        f5 = -x1 * x5 + 5 * np.tanh(20 * (x2 - x4))
        
        f = torch.stack([f1, f2, f3, f4, f5])
        dynamics = f + control_input
        
        return dynamics, f
    
    @staticmethod
    def target_dynamics_torch(time):
        """
        Compute desired trajectory dynamics using PyTorch
        """
        xd1Dot = torch.sin(2 * time)
        xd2Dot = -torch.cos(time)
        xd3Dot = torch.sin(3 * time) + torch.cos(-2 * time)
        xd4Dot = torch.sin(time) - torch.cos(-0.5 * time)
        xd5Dot = torch.sin(-time)
        
        return torch.stack([xd1Dot, xd2Dot, xd3Dot, xd4Dot, xd5Dot])

class PyTorchState:
    """PyTorch implementation of the State class"""
    def __init__(self, num_states, time_steps, device=None):
        # Force CPU usage
        device = 'cpu'
        self.num_states = num_states
        self.time_steps = time_steps
        self.device = device
        self.positions = torch.zeros((num_states, time_steps), device=device)
        self.positions[:, 0] = torch.tensor([0, -1, 3, -3, 3], device=device, dtype=torch.float32)
        self.velocities = torch.zeros((num_states, time_steps), device=device)
    
    def update_dynamics(self, step, control_input, time_step_delta, current_time):
        """Update state dynamics and return true f vector"""
        dynamics, f_true = PyTorchThermoDynamics.state_dynamics_torch(
            self.positions[:, step - 1], control_input, current_time
        )
        self.velocities[:, step] = dynamics
        # Euler integration
        self.positions[:, step] = self.positions[:, step - 1] + time_step_delta * self.velocities[:, step]
        return f_true

class PyTorchDesiredTrajectory:
    """PyTorch implementation of the DesiredTrajectory class"""
    def __init__(self, num_states, time_steps, device=None):
        # Force CPU usage
        device = 'cpu'
        self.num_states = num_states
        self.time_steps = time_steps
        self.device = device
        self.positions = torch.zeros((num_states, time_steps), device=device)
        self.velocities = torch.zeros((num_states, time_steps), device=device)
    
    def update_dynamics(self, step, time_step_delta, current_time):
        """Update desired trajectory dynamics"""
        self.velocities[:, step] = PyTorchThermoDynamics.target_dynamics_torch(
            torch.tensor(current_time, device=self.device, dtype=torch.float32)
        )
        # Euler integration
        self.positions[:, step] = self.positions[:, step - 1] + time_step_delta * self.velocities[:, step]

class PyTorchThermodynamicController:
    """
    PyTorch implementation of the thermodynamic neural network controller
    """
    def __init__(self, config, mu_case=1, device=None):
        # Force CPU usage
        device = 'cpu'
        self.device = device
        self.config = config
        self.mu_case = mu_case
        # Control parameters
        self.ke = config['ke']
        self.time_step_delta = config['time_step_delta']
        self.learning_rate = config['learning_rate']
        self.forgetting_factor = config['forgetting_factor']
        self.kT = config['kT']
        self.thetabar = config['thetabar']
        # Neural network
        self.dnn = ThermodynamicNeuralNetwork(
            num_inputs=config['num_inputs'],
            num_outputs=config['num_outputs'],
            num_layers=config['num_layers'],
            num_neurons=config['num_neurons']
        ).to(device)
        # Jacobian computer
        self.jacobian_computer = JacobianComputer(self.dnn)
        # Random seed for reproducibility
        torch.manual_seed(0)
        print(f"PyTorch ThermoDNN parameters: {self.count_parameters()} (device: {self.device})")
    
    def count_parameters(self):
        """Count total number of parameters in the network"""
        return sum(p.numel() for p in self.dnn.parameters())
    
    def calculate_mu_and_gradient(self, positions, weights, tracking_error):
        """
        Calculate drift term mu and its gradient based on mu_case
        """
        num_weights = len(weights)
        num_inputs = self.config['num_inputs']
        
        if self.mu_case == 1:
            # Zero drift case
            mu = torch.zeros(num_inputs, device=self.device)
            gradient_of_mu = torch.zeros((num_weights, num_inputs), device=self.device)
        
        elif self.mu_case == 2:
            # Tracking-based drift
            mu = 9 * tracking_error 
            gradient_of_mu = torch.zeros((num_weights, num_inputs), device=self.device)
        
        elif self.mu_case == 3:
            # Position-based drift
            pos_norm_sq = torch.norm(positions)**2
            mu = tracking_error * (0.01 * pos_norm_sq + 9)
            gradient_of_mu = torch.zeros((num_weights, num_inputs), device=self.device)
        
        elif self.mu_case == 4:
            # Weight-based drift
            weight_norm_sq = torch.norm(weights)**2
            mu = tracking_error * (0.01 * weight_norm_sq + 9)
            # Gradient computation
            # Ensure tracking_error and weights are 1D tensors
            tracking_error_flat = tracking_error.view(-1)
            weights_flat = weights.view(-1)
            # Compute outer product for correct shape: (num_weights, num_inputs)
            gradient_of_mu = 2 * 0.005 * torch.ger(weights_flat, tracking_error_flat)
        
        else:
            raise ValueError(f"Unknown mu_case: {self.mu_case}")
        
        return mu, gradient_of_mu
    
    def projection_operator(self, gradient, weights, theta_bar):
        """
        Projection operator to enforce weight bounds
        """
        weight_norm = torch.norm(weights)
        if weight_norm <= theta_bar:
            return gradient
        else:
            # Project gradient to maintain constraint
            projection_factor = 1.0 - (theta_bar / weight_norm)
            projected_gradient = gradient - projection_factor * (torch.dot(gradient, weights) / weight_norm**2) * weights
            return projected_gradient
    
    @torch.no_grad()
    def step(self, state_positions, tracking_error, desired_velocity, step_num):
        """
        Perform one control step with neural network weight updates
        """
        # Convert inputs to tensors
        state_tensor = torch.tensor(state_positions, device=self.device, dtype=torch.float32)
        tracking_error_tensor = torch.tensor(tracking_error, device=self.device, dtype=torch.float32)
        desired_velocity_tensor = torch.tensor(desired_velocity, device=self.device, dtype=torch.float32)
        
        # 1. Get NN output with current weights
        self.dnn.eval()
        nn_output = self.dnn(state_tensor.unsqueeze(0)).squeeze()
        
        # 2. Compute Jacobian for weight updates
        jacobian = self.jacobian_computer.compute(state_tensor)
        
        # 3. Get current weights as flat vector
        current_weights = torch.nn.utils.parameters_to_vector(self.dnn.parameters())
        
        # 4. Calculate mu and its gradient
        mu, gradient_of_mu = self.calculate_mu_and_gradient(state_tensor, current_weights, tracking_error_tensor)
        
        # 5. Compute weight update using thermodynamic learning rule
        loss = tracking_error_tensor
        num_weights = len(current_weights)
        
        # Brownian motion term
        dw = torch.randn(num_weights, device=self.device) * np.sqrt(self.time_step_delta)
        
        # Drift term computation
        drift_term1 = jacobian.T @ loss  # Gradient descent term
        drift_term2 = -self.forgetting_factor * current_weights  # Forgetting term
        
        # Thermodynamic term (if gradient_of_mu is available)
        if torch.any(gradient_of_mu != 0):
            drift_term3 = 0.5 * (1 + num_weights) * self.learning_rate * self.kT * (gradient_of_mu @ loss)
        else:
            drift_term3 = torch.zeros_like(current_weights)
        
        drift = self.learning_rate * (drift_term1 + drift_term2 + drift_term3)
        
        # Diffusion term
        diffusion_coeff = self.learning_rate * torch.sqrt(self.kT * torch.abs(torch.dot(loss, mu)))
        diffusion = diffusion_coeff * dw
        
        # Apply projection operator
        drift_projected = self.projection_operator(drift, current_weights, self.thetabar)
        diffusion_projected = self.projection_operator(diffusion, current_weights, self.thetabar)
        
        # Weight update
        weight_update = drift_projected * self.time_step_delta + diffusion_projected
        new_weights = current_weights + weight_update
        
        # Update network parameters
        torch.nn.utils.vector_to_parameters(new_weights, self.dnn.parameters())
        
        # 6. Compute control input
        control_input = desired_velocity_tensor - self.ke * tracking_error_tensor - nn_output
        
        return control_input.cpu().numpy(), nn_output.cpu().numpy(), mu.cpu().numpy()

    def evaluate_off_trajectory(self, x_rand_dataset):
        """
        Evaluate network performance on off-trajectory data points
        """
        errors = []
        self.dnn.eval()
        
        with torch.no_grad():
            for i in range(x_rand_dataset.shape[1]):
                x_rand = x_rand_dataset[:, i]
                
                # True function value
                x_rand_tensor = torch.tensor(x_rand, device=self.device, dtype=torch.float32)
                _, f_true = PyTorchThermoDynamics.state_dynamics_torch(x_rand_tensor)
                
                # Network approximation
                f_hat = self.dnn(x_rand_tensor.unsqueeze(0)).squeeze()
                
                # Error computation
                error_norm = torch.norm(f_true - f_hat).item()
                errors.append(error_norm)
        
        # RMS error
        rms_error = np.sqrt(np.mean(np.square(errors)))
        return rms_error

class PyTorchSimulation:
    """
    Main simulation class using PyTorch for thermodynamic neural network control
    """
    def __init__(self, config_path='config.json', device=None):
        # Force CPU usage
        device = 'cpu'
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.device = device
        print(f"PyTorchSimulation initialized with device: {self.device}")
        self.final_time = self.config['final_time']
        self.time_step_delta = self.config['time_step_delta']
        self.time_steps = int(self.final_time / self.time_step_delta)
        self.num_states = self.config['num_states']
        # Generate fixed random dataset for off-trajectory evaluation
        self.num_test_points = 90
        np.random.seed(42)
        self.X_rand = 0.5 * np.random.rand(self.num_states, self.num_test_points)
        # Storage for results
        self.results = {}
    
    def run_for_case(self, mu_case):
        """Run simulation for a specific mu_case"""
        print(f"\n--- Running PyTorch Simulation for Case {mu_case} ---")
        
        # Initialize state and trajectory
        state = PyTorchState(self.num_states, self.time_steps, self.device)
        desired_trajectory = PyTorchDesiredTrajectory(self.num_states, self.time_steps, self.device)
        
        # Initialize controller
        controller = PyTorchThermodynamicController(self.config, mu_case, self.device)
        
        # Storage for case-specific results
        state_errors = []
        f_approx_errors = []
        off_trajectory_errors = []
        
        # Simulation loop
        for step in range(1, self.time_steps):
            progress = (step / (self.time_steps - 1)) * 100
            print(f"\rProgress: {progress:.2f}%", end="", flush=True)
            current_time = step * self.time_step_delta
            
            # Compute tracking error
            tracking_error = (state.positions[:, step - 1] - desired_trajectory.positions[:, step - 1]).cpu().numpy()
            
            # Controller step
            control_input, nn_output, mu = controller.step(
                state.positions[:, step - 1].cpu().numpy(),
                tracking_error,
                desired_trajectory.velocities[:, step - 1].cpu().numpy(),
                step
            )
            
            # Update dynamics
            control_tensor = torch.tensor(control_input, device=self.device, dtype=torch.float32)
            f_true = state.update_dynamics(step, control_tensor, self.time_step_delta, current_time)
            desired_trajectory.update_dynamics(step, self.time_step_delta, current_time)
            
            # Calculate errors
            state_error = np.linalg.norm(tracking_error)
            f_approx_error = np.linalg.norm(f_true.cpu().numpy() - nn_output)
            
            state_errors.append(state_error)
            f_approx_errors.append(f_approx_error)
            
            # Periodic off-trajectory evaluation
            if step % 500 == 0:
                off_traj_error = controller.evaluate_off_trajectory(self.X_rand)
                off_trajectory_errors.append(off_traj_error)
                #print(f"Step {step}/{self.time_steps}: State Error = {state_error:.6f}, "
                #      f"F Approx Error = {f_approx_error:.6f}, Off-Traj Error = {off_traj_error:.6f}")
        print(f"\rCase {mu_case} completed!{' ' * 20}") 
        # Store results
        self.results[mu_case] = {
            'state_errors': np.array(state_errors),
            'f_approx_errors': np.array(f_approx_errors),
            'off_trajectory_errors': np.array(off_trajectory_errors),
            'final_state_error': state_errors[-1],
            'final_f_approx_error': f_approx_errors[-1],
            'final_off_traj_error': off_trajectory_errors[-1] if off_trajectory_errors else 0.0
        }
        
        #print(f"Case {mu_case} completed!")
        #rint(f"Final state error: {state_errors[-1]:.6f}")
        #print(f"Final function approximation error: {f_approx_errors[-1]:.6f}")
        
        return self.results[mu_case]
    
    def run_all_cases(self):
        """Run simulation for all 4 mu cases"""
        print("="*80)
        print("PYTORCH THERMODYNAMIC NEURAL NETWORK SIMULATION")
        print("5-state nonlinear control system with online learning")
        print("="*80)
        
        for mu_case in range(1, 5):
            self.run_for_case(mu_case)
        
        self.plot_comparative_results()
        self.print_summary()
    
    def plot_comparative_results(self):
        """Plot comparative results for all cases"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PyTorch Thermodynamic Neural Network Control Results', fontsize=16)
        
        time_vector = np.arange(1, self.time_steps) * self.time_step_delta
        
        # Plot 1: State tracking errors
        ax1 = axes[0, 0]
        for mu_case in range(1, 5):
            if mu_case in self.results:
                ax1.semilogy(time_vector, self.results[mu_case]['state_errors'], 
                         label=f'Case {mu_case}', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('State Tracking Error')
        ax1.set_title('State Tracking Error Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Function approximation errors
        ax2 = axes[0, 1]
        for mu_case in range(1, 5):
            if mu_case in self.results:
                ax2.semilogy(time_vector, self.results[mu_case]['f_approx_errors'], 
                         label=f'Case {mu_case}', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Function Approximation Error')
        ax2.set_title('Function Approximation Error Comparison')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Final errors comparison
        ax3 = axes[1, 0]
        cases = list(self.results.keys())
        final_state_errors = [self.results[case]['final_state_error'] for case in cases]
        final_f_errors = [self.results[case]['final_f_approx_error'] for case in cases]
        
        x_pos = np.arange(len(cases))
        width = 0.35
        
        ax3.bar(x_pos - width/2, final_state_errors, width, label='State Error', alpha=0.8)
        ax3.bar(x_pos + width/2, final_f_errors, width, label='Function Approx Error', alpha=0.8)
        ax3.set_xlabel('Case')
        ax3.set_ylabel('Final Error')
        ax3.set_title('Final Error Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Case {case}' for case in cases])
        ax3.legend()
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Plot 4: Off-trajectory performance
        ax4 = axes[1, 1]
        for mu_case in range(1, 5):
            if mu_case in self.results and len(self.results[mu_case]['off_trajectory_errors']) > 0:
                off_traj_steps = np.arange(len(self.results[mu_case]['off_trajectory_errors'])) * 500 * self.time_step_delta
                ax4.semilogy(off_traj_steps, self.results[mu_case]['off_trajectory_errors'], 
                         'o-', label=f'Case {mu_case}', linewidth=2, markersize=6)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Off-Trajectory RMS Error')
        ax4.set_title('Generalization Performance')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # In the PyTorchSimulation class...

    def print_summary(self):
        """Print simulation summary in a formatted table."""
        print("\n" + "="*80)
        print("PYTORCH THERMODYNAMIC NEURAL NETWORK SIMULATION SUMMARY")
        print("="*80)

        # Prepare data for the table
        # We use a dictionary where keys are column headers
        table_data = {
            'Metric': [
                'Final State Tracking Error',
                'Final Function Approx. Error',
                'Final Off-Trajectory Error'
            ]
        }

        # Populate the data for each case
        for mu_case in sorted(self.results.keys()):
            result = self.results[mu_case]
            # Add a new column for each case
            table_data[f'Case {mu_case}'] = [
                f"{result['final_state_error']:.6f}",
                f"{result['final_f_approx_error']:.6f}",
                f"{result['final_off_traj_error']:.6f}"
            ]

        # --- NEW: Use pandas to create and print a formatted table ---
        try:
            # We need to import pandas here if it's not globally available
            # But based on your original code, it should be.
            import pandas as pd

            # Configure pandas for better console display
            pd.set_option('display.max_colwidth', 40)
            pd.set_option('display.width', 120)

            df = pd.DataFrame(table_data)
            # Use the 'Metric' column as the table's index (row labels)
            df.set_index('Metric', inplace=True)
            
            print(df)

        except ImportError:
            print("Could not import pandas. Please install it (`pip install pandas`) for a formatted table.")
            # Fallback to the old printing method if pandas is not available
            for mu_case in sorted(self.results.keys()):
                result = self.results[mu_case]
                print(f"\nCase {mu_case}:")
                print(f"  Final state tracking error:       {result['final_state_error']:.6f}")
                print(f"  Final function approximation error: {result['final_f_approx_error']:.6f}")
                print(f"  Final off-trajectory error:         {result['final_off_traj_error']:.6f}")
        
        # Find and print the best performing case
        best_case = min(self.results.keys(), 
                        key=lambda x: self.results[x]['final_state_error'])
        print(f"\nBest performing case based on State Tracking Error: Case {best_case}")
        print("="*80)

def main():
    """Main function to run the PyTorch thermodynamic neural network simulation"""
    # Force CPU usage
    device = 'cpu'
    print("Using device: cpu")
    # Run simulation
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    simulation = PyTorchSimulation(config_path, device)
    simulation.run_all_cases()

if __name__ == "__main__":
    main()
