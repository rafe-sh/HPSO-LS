# HPSO-LS

HPSO-LS is a feature subset selection algorithm that combines Hybrid Particle Swarm Optimization (PSO) with a Local Search strategy. This repository provides the implementation of the HPSO-LS algorithm.

## Features

- Integration of PSO and Local Search for efficient feature subset selection.
- Customizable algorithm parameters to adapt to different optimization goals.
- Compatibility with popular machine learning frameworks and libraries.
- Visualization and analysis of results for better understanding and evaluation.
- Comprehensive documentation and code examples for easy implementation.


## Dataset

The HPSO-LS algorithm uses the "Sonar, Mines vs. Rocks" dataset for demonstration purposes. The dataset is included in the repository and can be found in the `sonar.mines` and `sonar.rocks` files. It consists of sonar signals and aims to discriminate between metal cylinders and rocks.

## Algorithm Parameters

The HPSO-LS algorithm provides various parameters that can be customized:

- Swarm size: Number of particles in the swarm.
- Maximum iterations: Maximum number of iterations for the optimization process.
- Cognitive coefficient: Weight for the particle's best-known position.
- Social coefficient: Weight for the global best-known position.
- Inertia weight: Controls the impact of the particle's previous velocity.
- Local search iterations: Number of iterations for the local search phase.

## Optimization Goals

The HPSO-LS algorithm aims to find the optimal feature subset that maximizes the performance of machine learning models. The optimization goals include:

- Maximizing classification accuracy.
- Minimizing training time.
- Reducing overfitting by selecting the most relevant features.


## Contact

For any questions or inquiries, please contact Rafe Sharif at [raffff81@gmail.com](mailto:raffff81@gmail.com).
