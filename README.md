# qmf: Quantum Mean Field

**qmf** is a Python package for implementing mean-field methods in quantum many-body physics. The package currently supports methods such as Hartree–Fock (HF) and Generalized Hartree–Fock (GHF) for both periodic and non–periodic systems, and it is designed to be easily extendable to incorporate additional mean–field techniques.

## Features

- **Hartree–Fock (HF):** Standard mean–field method for obtaining self–consistent field solutions. It support both:
    - **Restricted Hartree–Fock (RHF):** For closed–shell systems where the spatial part of the orbitals is identical for both spins.
    - **Unrestricted Hartree–Fock (UHF):** For open–shell systems, allowing different spatial orbitals for different spins.
- **Generalized Hartree–Fock (GHF):** Extension of HF that can incorporate pairing (anomalous) terms and break symmetries. This is suitable for study superconductivity. (testing)

## Future Development

Several functions are considered to be added in the future. Include but no limited with the followings. Welcome to comtribute!

- **Time–Dependent Hartree–Fock (TDHF):** For studying dynamical properties and excited states.
- **Spin–Density Functional Mean Field:** For systems where spin polarization is essential.
- **Slave Boson or Gutzwiller Mean Field Methods:** For strongly correlated electron systems.

## Installation

You can clone the repository and try a few examples of **qmf**:

```bash
git clone https://github.com/floatingCatty/qmf.git
