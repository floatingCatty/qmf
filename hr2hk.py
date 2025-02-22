import numpy as np
from utils.constants import anglrMId, norb_dict
from typing import Tuple, Union, Dict
from data.transforms import OrbitalMapper
from data import AtomicDataDict
from data.AtomicDataDict import with_edge_vectors
from utils.tools import float2comlex
from scipy.linalg import block_diag


class HR2HK:
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            spin_deg: bool=True,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            dtype=np.float32,
            ):

        self.dtype = dtype
        self.overlap = overlap
        self.ctype = float2comlex(self.dtype)
        self.spin_deg = spin_deg

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", spin_deg=spin_deg)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            assert idp.method == "e3tb", "The method of idp should be e3tb."
            self.idp = idp

        # get mask_to_basis incase that spin is neglected
        mask_to_basis = self.idp.mask_to_basis.copy()
        if self.spin_deg:
            mask_to_basis = mask_to_basis[...,None].repeat(2,2).reshape(mask_to_basis.shape[0], -1)
        self.mask_to_basis = mask_to_basis

        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()
        self.idp.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def __call__(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

        # for gGA mapping, there are two circumstances, one is spin-deg, including soc, in this case, the physical system does not have spin degree of freedom
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        atom_types = data.get(AtomicDataDict.ATOM_TYPE_KEY)
        if atom_types is None:
            data = self.idp(data)
            atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        atom_types = atom_types.flatten()

        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        norb = self.idp.full_basis_norb
        bondwise_hopping = np.zeros((len(orbpair_hopping), norb, norb), dtype=self.dtype)

        onsite_block = np.zeros((len(atom_types), norb, norb,), dtype=self.dtype)
        kpoints = data[AtomicDataDict.KPOINT_KEY]

        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        if isinstance(soc, np.ndarray):
            soc = soc.all()
        if soc:
            assert self.spin_deg, "SOC is only implemented for spin-degenerate case."
            if self.overlap:
                raise NotImplementedError("Overlap is not implemented for SOC.")
            
            orbpair_soc = data[AtomicDataDict.NODE_SOC_KEY]
            soc_upup_block = np.zeros((len(atom_types), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype)
            soc_updn_block = np.zeros((len(atom_types), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype)

        ist = 0
        for i,io in enumerate(self.idp.full_basis):
            jst = 0
            iorb = self.idp.flistnorbs[i] * self.idp.spin_factor
            for j,jo in enumerate(self.idp.full_basis):
                jorb = self.idp.flistnorbs[j] * self.idp.spin_factor
                orbpair = io+"-"+jo
                
                # constructing hopping blocks
                if io == jo:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                    onsite_block[:,ist:ist+iorb,jst:jst+jorb] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, iorb, jorb)
                # constructing onsite blocks
                if not self.overlap:
                    if soc and i==j:
                        soc_updn_tmp = orbpair_soc[:,self.idp.orbpair_soc_maps[orbpair]].reshape(-1, iorb, 2*jorb)
                        soc_upup_block[:,ist:ist+iorb,jst:jst+jorb] = soc_updn_tmp[:, :iorb,:jorb]
                        soc_updn_block[:,ist:ist+iorb,jst:jst+jorb] = soc_updn_tmp[:, :iorb,jorb:]
                
                jst += jorb
            ist += iorb
        
        # address the spin degeneracy
        if self.spin_deg:
            shape = list(onsite_block.shape)
            onsite_block = np.stack([onsite_block, np.zeros_like(onsite_block), np.zeros_like(onsite_block), onsite_block], axis=-1).reshape(shape+[2,2]).transpose(0,1,3,2,4)

            if soc:
                onsite_block[:,:,0,:,0] = onsite_block[:,:,0,:,0] + 0.5 * soc_upup_block
                onsite_block[:,:,1,:,1] = onsite_block[:,:,1,:,1] + 0.5 * soc_upup_block.conj()
                onsite_block[:,:,0,:,1] = 0.5 * soc_updn_block
                onsite_block[:,:,1,:,0] = 0.5 * soc_updn_block.conj()

            onsite_block = onsite_block.reshape(-1, onsite_block.shape[1]*2, onsite_block.shape[3]*2)
            shape = list(bondwise_hopping.shape)
            bondwise_hopping = np.stack([bondwise_hopping, np.zeros_like(bondwise_hopping), np.zeros_like(bondwise_hopping), bondwise_hopping], axis=-1).reshape(shape+[2,2]).transpose(0,1,3,2,4)
            bondwise_hopping = bondwise_hopping.reshape(-1, bondwise_hopping.shape[1]*2, bondwise_hopping.shape[3]*2)

        self.onsite_block = {}
        self.bondwise_hopping = {}
        onsite_tkR = {}
        hopping_tkR = {}
        norb = 0
        for sym, at in self.idp.chemical_symbol_to_type.items():
            mask = self.mask_to_basis[at]
            atmask = atom_types.flatten() == at
            self.onsite_block[sym] = onsite_block[atmask][:,mask][:,:,mask]
            norb += self.onsite_block[sym].shape[1] * self.onsite_block[sym].shape[0]
        
        for bsym, bt in self.idp.bond_to_type.items():
            sym1, sym2 = bsym.split("-")
            at1, at2 = self.idp.chemical_symbol_to_type[sym1], self.idp.chemical_symbol_to_type[sym2]
            mask1 = self.mask_to_basis[at1]
            mask2 = self.mask_to_basis[at2]
            btmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() == bt
            self.bondwise_hopping[bsym] = bondwise_hopping[btmask][:,mask1][:,:,mask2]
        
        # R2K procedure can be done for all kpoint at once.
        # from now on, any spin degeneracy have been removed. All following blocks consider spin degree of freedom
        block = np.zeros((kpoints.shape[0], norb, norb), dtype=self.ctype)

        atom_id_to_indices = {}
        atom_id_to_indices_phy = {}
        ist = 0
        type_count = [0] * len(self.idp.type_names)
        for i, at in enumerate(atom_types.flatten()):
            idx = type_count[at]
            sym = self.idp.type_names[at]
            oblock = self.onsite_block[sym][idx]
            block[:,ist:ist+oblock.shape[0],ist:ist+oblock.shape[1]] = oblock[None,...]
            atom_id_to_indices[i] = slice(ist, ist+oblock.shape[0])
            ist += oblock.shape[0]

            type_count[at] += 1


        cell_inv = np.linalg.inv(data[AtomicDataDict.CELL_KEY])
        for bsym, btype in self.idp.bond_to_type.items():
            bmask = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() == btype
            shifts_vec = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][bmask]
            for i, edge in enumerate(edge_index.T[bmask]):
                iatom = edge[0]
                jatom = edge[1]
                shifts_in_vec = data[AtomicDataDict.POSITIONS_KEY][jatom] - data[AtomicDataDict.POSITIONS_KEY][jatom]
                shifts_in_vec = cell_inv @ shifts_in_vec
                iatom_indices = atom_id_to_indices[int(iatom)]
                jatom_indices = atom_id_to_indices[int(jatom)]
                hblock = self.bondwise_hopping[bsym][i]
                block[:,iatom_indices,jatom_indices] += hblock[None,...].astype(block.dtype) * \
                    np.exp(1j * 2 * np.pi * (kpoints @ (shifts_vec[i]+shifts_in_vec))).reshape(-1,1,1)

        block = block + block.transpose(0,2,1).conj()
        block = np.ascontiguousarray(block)

        data[self.out_field] = block

        return data # here output hamiltonian have their spin orbital connected between each other