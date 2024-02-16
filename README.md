# Towards Training Without Depth Limits: Large Batch Normalization Without Gradient Explosion

![AMAL_PROJECT](https://github.com/Allaa-boutaleb/Infinitely-Deep-MLPs/assets/60470207/580961d6-3446-4c98-a262-daf93b0d4cfa)

Link to poster: [AMAL_PROJECT.pdf](https://github.com/Allaa-boutaleb/Infinitely-Deep-MLPs/files/14315814/AMAL_PROJECT.pdf)


## Introduction

This repository is dedicated to the exploration and implementation of the groundbreaking research introduced in the paper titled "Towards Training Without Depth Limits: Large Batch Normalization Without Gradient Explosion". This study represents a significant advancement in the field of deep learning, focusing on overcoming the challenges associated with depth limits in neural network training through innovative batch normalization techniques.

### Authors of the Original Paper
- Alexandru Meterez
- Amir Joudaki
- Francesco Orabona
- Alexander Immer
- Gunnar Rätsch
- Hadi Daneshmand

**Contact:** [Alexandru Meterez](mailto:alexandrumeterez@gmail.com)

### Contributors to This Repository
- Allaa BOUTALEB
- Samy NEHLIL
- Ghiles OUHENIA

**Disclaimer:** The work in this repository is based on the original paper by the authors listed above. We have simply studied the paper, reproduced the results, refactored the code, and created a poster summarizing the research. All intellectual property rights belong to the original authors.

## Repository Structure

This repository is organized to provide a comprehensive toolkit for replicating the experiments and understanding the concepts introduced in the paper. Below is an overview of the repository's structure:

- `src/`: Contains the core code, including:
  - `model.py`: Definitions for the MLP with custom Batch Normalization and Gain Activation.
  - `utils.py`: Utility functions for training and testing.
  - `constants.py`: Constants used throughout the codebase.
- `Notebooks/`: Jupyter notebooks for experimental replication and result visualization:
  - `Experiments.ipynb`: Demonstrates the Isometry Gap, Gain Effect, Weight Initialization, and Gain Exponent's impact on gradient explosion.
  - `Train.ipynb`: Reproduces the main training/testing accuracy plots on the MNIST dataset using the proposed neural network architecture.
- `Figures/`: Contains all generated figures in PDF format.
- `training/`: Stores training artifacts:
  - `checkpoints/`: Model checkpoints for interrupting and resuming training loops.
  - `results/`: `.csv` files with training and testing accuracy results.


## Accessing the Paper

For an in-depth exploration of the concepts and methodologies presented in this research, access the paper [here](https://arxiv.org/abs/2310.02012).

## License

This project is open-sourced under the MIT license. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Our sincere gratitude goes to the original authors of the paper for their groundbreaking contributions to the field of deep learning. This repository serves as a homage to their work, aiming to facilitate further research and exploration in the community.
