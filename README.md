# Neural ODEs for the FitzHugh-Nagumo Model

This repository contains MATLAB scripts implementing Neural Ordinary Differential Equations (Neural ODEs) to learn and simulate the **FitzHugh-Nagumo** dynamic model. Three different numerical solvers are explored and compared within this framework:

- **Runge-Kutta 4th Order (RK4)**
- **TR-BDF2 (Trapezoidal Rule - Backward Differentiation Formula 2)**
- **I-TR-BDF2 (Improved TR-BDF2)**

The primary goal of this work is to investigate the performance and stability of these solvers in learning the FitzHugh-Nagumo dynamics using Neural ODE structures.

## Implementation Notes

- The core training pipeline is inspired by and adapted from the repository:  
  🔗 [https://github.com/mldiego/neuralODE](https://github.com/mldiego/neuralODE)

- All training is performed in MATLAB, and the scripts are structured to allow easy switching between integration methods.

- The models were trained with synthetic data generated from the FitzHugh-Nagumo system and evaluated based on trajectory prediction accuracy and error metrics.

## Citation

If you find this work useful, please cite the original [Neural ODE GitHub repository](https://github.com/mldiego/neuralODE) as the basis for this extension.

## License

This repository inherits the license of the original neuralODE repo unless otherwise specified.
