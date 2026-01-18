This repository is for the development of a C++ extension for a pythonic implementation of the radial fingerprints used in the Rapid Artificial Neural Network of Dr. Kip Barrett and Dr. Doyl Dickel's work here: https://www.sciencedirect.com/science/article/abs/pii/S0927025621002068


Simple pair interactions are considered and summed over all neighbors of a given atom.
The pair-interaction structural fingerprint for atom $\alpha$ is defined as:

![05FD32D8-F999-44BE-9990-573F88FFE696_4_5005_c](https://github.com/user-attachments/assets/19f2de77-49ce-4be4-ad4c-0eb3a73e03ac)


where $\beta$ labels neighboring atoms within the cutoff radius of atom $\alpha$.

The cutoff radius $r_c$ is defined using a MEAM-based piecewise function:

![AA886FD5-5DD8-4D32-9272-D8CF20905678_4_5005_c](https://github.com/user-attachments/assets/3a154b87-a53f-4395-8623-a45d454b28e2)


The cutoff smoothly transitions from 1 to 0 to account for negligible contributions from atoms at large separations. This ensures locality while maintaining differentiability for learning-based interatomic potentials.

