# Chignolin: a 166-atom mini-protein dataset

This dataset consists of 9543 conformations that sampled by replica exchange MD and calculated at DFT level by Gaussian 16. The raw folder contains a npz file including id, R, Z, N, F, E.

- ```id``` is an unique number of the conformation of Chignolins. (Shape: (Nsample,))

- ```R``` is the coordiante of the conformation of Chignolins. (Shape: (Nsample * Natoms, 3)) (Unit: Å)

- ```Z``` is the atomic number of Chignolins. (Shape: (Nsample * Natoms,))

- ```N``` is the number of atoms in Chignolins. (Shape: (Nsample,))

- ```F``` is the negative forces (engrads) of the conformation of Chignolins. (Shape: (Nsample * Natoms, 3)) (Unit: Hartree/Bohr)

- ```E``` is the potential energy of the conformation of Chignolins. (Shape: (Nsample,)) (Unit: Hartree)

The processed folder contains a pre-processed data for model input. The unit of enengy and forces are eV and eV/Å.

*NOTE:* The potential energies in processed folder are substracted by corresponding reference energy.