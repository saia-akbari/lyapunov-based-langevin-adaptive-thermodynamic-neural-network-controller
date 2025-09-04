[Read Me.txt](https://github.com/user-attachments/files/22141936/Read.Me.txt)
This repository contains the official implementation of the LyLA-Therm model, a Lyapunov-based Langevin Adaptive Thermodynamic Neural Network Controller, as presented in the paper "LyLA-Therm: Lyapunov-based Langevin Adaptive Thermodynamic Neural Network Controller".

LyLA-Therm is a novel control method that enhances traditional Lyapunov-based Deep Neural Network (DNN) controllers by incorporating principles from thermodynamics and statistical mechanics. Inspired by the Langevin equation, we've developed a unique update law for the controller's neural network weights, which takes the form of a stochastic differential equation (SDE).

This approach elegantly balances the exploration-exploitation trade-off in control systems:
Exploitation: A drift term is meticulously designed to minimize the system's generalized internal energy, driving the system towards stable states.
Exploration: A diffusion term, governed by a user-defined generalized temperature law, introduces controlled stochastic noise, enabling the system to explore its state space and escape local minima.
By adjusting the temperature law, LyLA-Therm allows for fine-tuned control over fluctuations, making it a powerful and adaptable solution for complex control problems.

This repository includes the code for the four simulations performed in the paper to demonstrate the effectiveness and behavior of LyLA-Therm compared to a baseline deterministic controller.

Baseline Lb-DNN: This simulation represents the performance of a standard Lyapunov-based DNN controller with a deterministic update law.

LyLA-Therm Architectures: These simulations implement the LyLA-Therm controller with three different designs of the temperature law. This showcases the controller's adaptability and the impact of the diffusion term on system performance.
