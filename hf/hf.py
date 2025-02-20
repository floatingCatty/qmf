import numpy as np
import copy
from hr2hk import HR2HK
from data import _keys, AtomicDataDict
from mixing import PDIIS, Linear
from typing import Dict, Union

class HartreeFock:
    def __init__(
        self,
        basis: Dict[str, Union[str, list]], # ep. {"Si": ["s", "p", "d"]}
        nocc: int,
        idx_intorb: Dict[str, list],
        intparams: Dict[str, dict],
        nspin: int=1,
        mixer_options: dict={},
        overlap: bool=False,
        dtype=np.float64
    ):
        self.mixer_options = mixer_options
        self.spin_deg = nspin < 2
        self.dtype= dtype
        self.overlap = overlap
        self.idx_intorb = idx_intorb
        self.intparams = intparams
        self.nocc = nocc

        
        self.hr2hk = HR2HK(
            basis=basis,
            spin_deg=self.spin_deg,
            edge_field=_keys.EDGE_FEATURES_KEY,
            node_field=_keys.NODE_FEATURES_KEY,
            out_field=_keys.HAMILTONIAN_KEY,
            overlap=False,
            dtype=dtype
        )

        if self.overlap:
            self.sr2sk = HR2HK(
                basis=basis,
                spin_deg=self.spin_deg,
                edge_field=_keys.EDGE_OVERLAP_KEY,
                node_field=_keys.NODE_OVERLAP_KEY,
                out_field=_keys.OVERLAP_KEY,
                overlap=True,
                dtype=dtype
            )
        
    def run(self, data: AtomicDataDict.Type, max_iter=500, tol=1e-6, kBT=1e-5, ntol=1e-4, verbose=True):
        
        # get the first trial of density matrix
        data = self.hr2hk(data)
        if self.overlap:
            data = self.sr2sk(data)
            sk = data[_keys.OVERLAP_KEY]
        else:
            sk = None

        hk = data[_keys.HAMILTONIAN_KEY]

        D, efermi = compute_density_matrices(F=hk, S=sk, nocc=self.nocc, kBT=kBT, ntol=ntol)

        if self.mixer_options.get("method") == "Linear":
            mixer = Linear(**self.mixer_options, p0=D)
        elif self.mixer_options.get("method") == "PDIIS":
            mixer = PDIIS(**self.mixer_options, p0=D)
        elif len(self.mixer_options)==0:
            mixer = None
        else:
            raise NotImplementedError("Mixer other than Linear/PDIIS have not been implemented.")
        
        indices = setup_indices(
            atom_type=data[_keys.ATOM_TYPE_KEY].flatten(),
            idp=self.hr2hk.idp,
            idx_intorb=self.idx_intorb
            )

        for iteration in range(max_iter):
            F = hk
            for sym, idxs in indices.items():
                for i, (upidx, dnidx) in enumerate(idxs):
                    F = add_interaction(h_mat=F, upidx=upidx, dnidx=dnidx, D=D, **self.intparams[sym][i])

            # Compute new density matrices
            D_new, efermi = compute_density_matrices(F=F, S=sk, nocc=self.nocc, kBT=kBT, efermi0=efermi, ntol=ntol)

            # Compute energy
            E_tot = compute_energy(hk, F, D_new)

            # Compute density difference
            d = np.linalg.norm(D_new - D)

            # scf_history.append((iteration, E_tot, d))

            if verbose:
                print(f"Iter {iteration:3d}: E = {E_tot:.6f}, ΔD = {d:.6e}")

            # Check convergence
            if d < tol:
                D = D_new.copy()
                if verbose:
                    print(f"Convergence acheived!")
                break

            # Mixing for stability
            D = mixer.update(D_new)

        else:
            print("WARNING: HF did not converge within max_iter!")

        data[_keys.REDUCED_DENSITY_MATRIX_KEY] = D
        data["efermi"] = efermi

        return data
    
    def update_Fock(self, data):
        # compute the density matrix from fock

        indices = setup_indices(
            atom_type=data[_keys.ATOM_TYPE_KEY].flatten(),
            idp=self.hr2hk.idp,
            idx_intorb=self.idx_intorb
            )

        data = self.hr2hk(data)
        hk = data[_keys.HAMILTONIAN_KEY]
        D = data[_keys.REDUCED_DENSITY_MATRIX_KEY]

        F = hk
        for sym, idxs in indices.items():
            for i, (upidx, dnidx) in enumerate(idxs):
                F = add_interaction(h_mat=F, upidx=upidx, dnidx=dnidx, D=D, **self.intparams[sym][i])
        
        data[_keys.HAMILTONIAN_KEY] = F

        return data

        
"""
    The interaction are set the same for the same chemical element. So we shoud be able to get a indices for each element
"""
def setup_indices(atom_type, idp, idx_intorb):
    """Return the list of impurity orbital indices."""
    # Impurity orbitals: first n_imp orbitals, for spin up and down
    idp.get_orbital_maps()
    norbtotal = idp.atom_norb[atom_type]
    if idp.spin_deg:
        norbtotal = norbtotal * 2
    start_idx = np.cumsum(np.concat(([0],norbtotal)))
    end_idx = start_idx[1:]
    start_idx = start_idx[:-1]
    indices = {}
    for sym in idx_intorb.keys():
        indices[sym] = []
        tp = idp.chemical_symbol_to_type[sym]
        for itorb in idx_intorb[sym]:
            maps = idp.orbital_maps[sym][idp.basis[sym][itorb]]
            s, e = maps.start, maps.stop
            if idp.spin_deg:
                s *= 2
                e *= 2
            upind = np.arange(s,e,2)
            dnind = np.arange(s+1,e+1,2)
            indices[sym].append([(start_idx[atom_type==tp][:,None] + upind[None,:]).reshape(-1), (start_idx[atom_type==tp][:,None] + dnind[None,:]).reshape(-1)])
    
    return indices
        

def add_interaction(h_mat, upidx, dnidx, D, U, J, Up, Jp):
    """
    Build the normal and anomalous mean-field potentials (Fock matrices).

    Parameters
    ----------
    D : ndarray
        Normal density matrix.
    P : ndarray
        Anomalous density matrix.

    Returns
    -------
    F : ndarray
        Normal Fock matrix.
    Delta : ndarray
        Anomalous Fock matrix.
    """
    norb = h_mat.shape[1]
    nk = h_mat.shape[0]

    F = np.zeros((nk, norb, norb), dtype=h_mat.dtype)

    # Get impurity indices

    # Hartree and Fock contributions from Slater-Kanamori
    # Only impurity orbitals have interactions
    const = 0

    F[:, upidx, upidx] += U * D[dnidx, dnidx].real[None,...]
    F[:, dnidx, dnidx] += U * D[upidx, upidx].real[None,...]

    const -= U * np.sum(D[dnidx, dnidx] * D[upidx, upidx])

    # F[up_imp, up_imp] += 0.5 * Up * D[up_imp, up_imp].sum().real + 0.5 * Up * D[dn_imp, dn_imp].sum().real
    # F[dn_imp, dn_imp] += 0.5 * Up * D[dn_imp, dn_imp].sum().real + 0.5 * Up * D[up_imp, up_imp].sum().real
    # F[up_imp, up_imp] -= 0.5 * Up * D[up_imp, up_imp].real + 0.5 * Up * D[dn_imp, dn_imp].real
    # F[dn_imp, dn_imp] -= 0.5 * Up * D[dn_imp, dn_imp].real + 0.5 * Up * D[up_imp, up_imp].real

    F[:, upidx, upidx] += Up * D[dnidx, dnidx].sum().real[None,...]
    F[:, dnidx, dnidx] += Up * D[upidx, upidx].sum().real[None,...]
    F[:, upidx, upidx] -= Up * D[dnidx, dnidx].real[None,...]
    F[:, dnidx, dnidx] -= Up * D[upidx, upidx].real[None,...]
    const += 0.5 * Up * np.sum(D[dnidx, dnidx] * D[upidx, upidx]) - 0.5 * Up * (D[dnidx, dnidx].sum()*D[upidx, upidx].sum())

    # off diagonal
    # F[np.ix_(up_imp, up_imp)] -= 0.5 * (Up-J) * D[np.ix_(up_imp, up_imp)]
    # F[np.ix_(dn_imp, dn_imp)] -= 0.5 * (Up-J) * D[np.ix_(dn_imp, dn_imp)]
    # F[up_imp, up_imp] += 0.5 * (Up-J) * D[up_imp, up_imp].real
    # F[dn_imp, dn_imp] += 0.5 * (Up-J) * D[dn_imp, dn_imp].real
    F[:, dnidx, upidx] += J * (D[upidx, dnidx] - D[upidx, dnidx].sum())[None,...]
    F[:, upidx, dnidx] += J * (D[dnidx, upidx] - D[dnidx, upidx].sum())[None,...]

    const += 0.5 * J * np.sum(D[upidx, dnidx] * D[dnidx, upidx]) - J * (D[upidx, dnidx].sum()*D[dnidx, upidx].sum())
    
    F[:,upidx[:,None], upidx[None,:]] += J * D[np.ix_(dnidx, dnidx)].T[None,...]
    F[:,dnidx[:,None], dnidx[None,:]] += J * D[np.ix_(upidx, upidx)].T[None,...]
    F[:,upidx, upidx] -= J * D[dnidx, dnidx].real[None,...]
    F[:,dnidx, dnidx] -= J * D[upidx, upidx].real[None,...]

    const += - J * (D[np.ix_(upidx, upidx)] * D[np.ix_(dnidx, dnidx)].T).sum() + J * (D[dnidx, dnidx].real * D[upidx, upidx].real).sum()

    # Add the one-body Hamiltonian
    F = F + h_mat

    # subtract the constant term
    # F[:, upidx, upidx] -= const
    # F[:, dnidx, dnidx] -= const

    return F

def diagonalize_fock(F):
    """
    Diagonalize the generalized Fock matrix.

    Parameters
    ----------
    F_GHF : ndarray
        Generalized Fock matrix.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues sorted in ascending order.
    eigvecs : ndarray
        Corresponding eigenvectors.
    """
    nk = F.shape[0]
    norb = F.shape[1]
    eigvals, eigvecs = np.linalg.eigh(F)
    # Sort eigenvalues and eigenvectors
    # idx = np.argsort(eigvals)
    # idx = idx + np.arange(nk)[:,None] * norb
    # eigvals = eigvals.reshape(-1)[idx.reshape[-1]].reshape(nk, norb)
    # eigvecs = eigvecs.transpose(0,2,1).reshape(-1,norb)[idx].reshape(nk, norb, norb).transpose(0,2,1)

    return eigvals, eigvecs

def compute_density_matrices(F, nocc, S=None, kBT=1e-5, efermi0=0., ntol=1e-4):
    """
    Compute the normal and anomalous density matrices from occupied states.

    Parameters
    ----------
    F : ndarray [nk, norb, norb]
        Fock matrix.
    nocc : int
        Number of electrons to occupy.

    Returns
    -------
    D_new : ndarray
        Updated normal density matrix.
    P_new : ndarray
        Updated anomalous density matrix.
    """
    # Occupy the lowest nocc states
    # In Nambu space, each quasiparticle state can hold 2 electrons
    # However, to keep it simple, assume nocc <= dim
    # and occupy the lowest nocc states

    # Occupy states with negative eigenvalues (assuming symmetric spectrum)
    # This is more accurate for superconducting systems
    # Fx=kSx
    # S = UU*, Fx=kUU*x, U^-1 F U*^-1 y=ky, y=U*x, x=U*-1 y "*" for dagger
    if F.ndim == 2:
        F = F[None,...]
    if S is not None and S.ndim == 2:
        S = S[None,...]
    
    if S is not None:
        chklowt = np.linalg.cholesky(S) # U
        chklowtinv = np.linalg.inv(chklowt)
        F = chklowtinv @ F @ np.transpose(chklowtinv,(0,2,1)).conj()

    efermi = find_E_fermi(F, nocc=nocc, kBT=kBT, E_fermi0=efermi0, ntol=ntol)
    eigvals, eigvecs = diagonalize_fock(F=F)
    eigvecs = eigvecs.transpose(0,2,1)
    if S is not None:
        eigvecs = np.einsum("kji, kni->knj", chklowtinv.conj().T, eigvecs)

    factor = fermi_dirac(eigvals, efermi, kBT)
    mask = factor > 1e-10
    factor = factor[mask]
    eigvecs = eigvecs[mask,:]

    D_new = (eigvecs * factor[...,None]).conj().T @ eigvecs
    D_new /= F.shape[0]
    # Ensure Hermiticity
    D_new = (D_new + D_new.T.conj()) / 2

    # assert np.abs(D_new.imag).max() < 1e-10, np.abs(D_new.imag).max()

    return D_new, efermi

def compute_energy(h_mat, F, D):
    """
    Compute the total energy.

    Parameters
    ----------
    F : ndarray
        Normal Fock matrix.
    Delta : ndarray
        Anomalous Fock matrix.
    D : ndarray
        Normal density matrix.
    P : ndarray
        Anomalous density matrix.

    Returns
    -------
    E_tot : float
        Total Hartree-Fock energy.
    """
    # Standard HF energy: E = 0.5 * Tr[D h] + 0.5 * Tr[D F]
    E_one_body = 0.5 * np.trace(D @ h_mat, axis1=1, axis2=2).real
    E_two_body = 0.5 * np.trace(D @ F, axis1=1, axis2=2).real
    E_tot = (E_one_body + E_two_body).mean()
    return E_tot


def compute_nocc(Hks, E_fermi: float, kBT: float):
    """
        Hks: [nk, norb, norb],
    """
    eigval = np.linalg.eigvalsh(Hks)
    counts = fermi_dirac(eigval, E_fermi, kBT)
    n = counts.sum() / Hks.shape[0]

    return n

def find_E_fermi(Hks, nocc, kBT: float=1e-4, E_fermi0: float=0, ntol: float=1e-8, max_iter=100.):
    E_up = E_fermi0 + 1
    E_down = E_fermi0 - 1
    nocc_converge = False

    nc = compute_nocc(Hks=Hks, E_fermi=E_fermi0, kBT=kBT)
    if abs(nc - nocc) < ntol:
        E_fermi = copy.deepcopy(E_fermi0)
        nocc_converge = True
    else:
        # first, adjust E_range
        E_range_converge = False
        while not E_range_converge:
            n_up = compute_nocc(Hks=Hks, E_fermi=E_up, kBT=kBT)
            n_down = compute_nocc(Hks=Hks, E_fermi=E_down, kBT=kBT)
            if n_up >= nocc >= n_down:
                E_range_converge = True
            else:
                if nocc > n_up:
                    E_up = E_fermi0 + (E_up - E_fermi0) * 2.
                else:
                    E_down = E_fermi0 + (E_down - E_fermi0) * 2.
        
        E_fermi = (E_up + E_down) / 2.

    # second, doing binary search  
    niter = 0

    while not nocc_converge and niter < max_iter:
        nc = compute_nocc(Hks=Hks, E_fermi=E_fermi, kBT=kBT)
        if abs(nc - nocc) < ntol:
            nocc_converge = True
        else:
            if nc < nocc:
                E_down = copy.deepcopy(E_fermi)
            else:
                E_up = copy.deepcopy(E_fermi)
            
            # update E_fermi
            E_fermi = (E_up + E_down) / 2.

        niter += 1

    if abs(nc - nocc) > ntol:
        raise RuntimeError("The Fermi Energy searching did not converge")

    return E_fermi

def fermi_dirac(energy, E_fermi, kBT):
    x = (energy - E_fermi) / kBT
    out = np.zeros_like(x)
    out[x < -500] = 1.
    mask = (x > -500) * (x < 500)
    out[mask] = 1./(np.exp(x[mask]) + 1)
    return out

def symsqrt(matrix): # this may returns nan grade when the eigenvalue of the matrix is very degenerated.
    """Compute the square root of a positive definite matrix."""
    # _, s, v = safeSVD(matrix)
    _, s, v = np.linalg.svd(matrix)
    v = v.conj().T

    good = s > s.max(axis=-1, keepdims=True) * s.shape[-1] * np.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.shape[-1]:
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, np.zeros((), dtype=s.dtype))
    
    shape = list(s.shape[:-1]) + [1] + [s.shape[-1]]
    return (v * np.sqrt(s).reshape(shape)) @ v.conj().swapaxes(-1,-2)


