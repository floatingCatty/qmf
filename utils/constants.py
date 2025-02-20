import numpy as np
import ase
from scipy.constants import Boltzmann, pi, elementary_charge, hbar

ALLOWED_VERSIONS = [1,2]
CUBIC_MAG_NUM_DICT = {'s':[0], 'p':[-1, 0, 1], 'd':[-2, -1, 0, 1, 2]}
LM_MAG_NUM_DICT    = {'s':[0], 'p':[-1, 0, 1], 'd':[-2, -1, 0, 1, 2]}


anglrMId = {'s':0,'p':1,'d':2,'f':3,'g':4,'h':5}
orbitalId = {0:'s',1:'p',2:'d',3:'f',4:'g',5:'h'}
anglrMId_r = {0:'s',1:'p',2:"d",3:"f",4:"g",5:"h"}

norb_dict = {'s':1,'e':2,'p':3,'q':4,'d':5,'j':6,'f':7,'o':8,'g':9,'h':11, "i":15, "k":21, "l": 25, "m": 35}
norb_dict_r = {1:'s', 2:'e', 3:'p', 4:'q', 5:'d', 6:'j', 7:'f', 8:'o', 9:'g', 11:'h', 15:"i", 21:"k", 25:"l", 35:"m"}

SKBondType = {0:'sigma',1:'pi',2:'delta'}
au2Ang = 0.529177249
Bohr2Ang = 0.529177249
# bond integral index in DFTB sk files. specific.
SKAnglrMHSID = {'dd':np.array([0,1,2]),
                'dp':np.array([3,4]), 'pd':np.array([3,4]),
                'pp':np.array([5,6]),
                'ds':np.array([7]),   'sd':np.array([7]),
                'ps':np.array([8]),   'sp':np.array([8]),
                'ss':np.array([9])}
h_all_types = ['ss','sp','sd','pp','pd','dd']

atomic_num_dict = ase.atom.atomic_numbers
atomic_num_dict_r = dict(zip(atomic_num_dict.values(), atomic_num_dict.keys()))
MaxShells  = 3
NumHvals   = 10

Orbital_Order_Wan_Default = { 's': ['s'],
                              'p': ['pz','px','py'],
                              'd': ['dz2','dxz','dyz','dx2-y2','dxy'],
                              'f': ['fz3', 'fxz2', 'fyz2', 'fxyz', 'fx3-3xy2','f3x2y-y3','fx2z-y2z'],
                              }
Orbital_Order_SK = {'s': ['s'],
                    'p': ['py','pz','px'],
                    'd': ['dxy','dyz','dz2','dxz','dx2-y2'],
                    'f': ['fz3', 'fxz2', 'fyz2', 'fxyz', 'fx3-3xy2','f3x2y-y3','fx2z-y2z']
                    }

m_dict = {"s": 0, "py": -1, "pz": 0, "px": 1, "dxy": -2, "dyz": -1, "dz2": 0, "dxz": 1, "dx2-y2": 2}


ABACUS_orbital_number_m = {
    "s": [0],
    "p": [0, 1, -1],
    "d": [0, 1, -1, 2, -2],
    "f": [0, 1, -1, 2, -2, 3, -3],
    "g": [0, 1, -1, 2, -2, 3, -3, 4, -4],
    "h": [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
}

DeePTB_orbital_number_m = {
    "s": [0],
    "p": [-1, 0, 1],
    "d": [-2, -1, 0, 1, 2],
    "f": [-3, -2, -1, 0, 1, 2, 3],
    "g": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    "h": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
}


ABACUS2DeePTB = {
            0: np.eye(1, dtype=np.float32),
            1: np.eye(3, dtype=np.float32)[[2, 0, 1]],
            2: np.eye(5, dtype=np.float32)[[4, 2, 0, 1, 3]],
            3: np.eye(7, dtype=np.float32)[[6, 4, 2, 0, 1, 3, 5]],
            4: np.eye(9, dtype=np.float32)[[8, 6, 4, 2, 0, 1, 3, 5, 7]],
            5: np.eye(11, dtype=np.float32)[[10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]]
        }

ABACUS2DeePTB[1][[0, 2]] *= -1
ABACUS2DeePTB[2][[1, 3]] *= -1
ABACUS2DeePTB[3][[0, 6, 2, 4]] *= -1
ABACUS2DeePTB[4][[1, 7, 3, 5]] *= -1
ABACUS2DeePTB[5][[0, 10, 8, 2, 6, 4]] *= -1

OPENMX2DeePTB = {
            "s": np.eye(1),
            "p": np.eye(3)[[1, 2, 0]],
            "d": np.eye(5)[[2, 4, 0, 3, 1]],
            "f": np.eye(7)[[6, 4, 2, 0, 1, 3, 5]]
        }

PYSCF_orbital_number_m = {
    "s": [0],
    "p": [1, -1, 0],
    "d": [-2, -1, 0, 1, 2],
    "f": [-3, -2, -1, 0, 1, 2, 3],
    "g": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    "h": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
}

PYSCF2DeePTB = {
            0: np.eye(1),
            1: np.eye(3)[[1, 2, 0]],   #1, -1, 0 -> -1, 0, 1
            2: np.eye(5),
            3: np.eye(7),
            4: np.eye(9),
            5: np.eye(11)
        }

# k = Boltzmann # k is the Boltzmann constant in old NEGF module
Coulomb = 6.24150974e18 # in the unit of eV*Angstrom
eV2J = 1.6021766208e-19 # in the unit of J