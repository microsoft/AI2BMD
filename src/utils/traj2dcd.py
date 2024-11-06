import MDAnalysis as mda
from ase.io import read
from MDAnalysis.coordinates.DCD import DCDWriter
import os
import argparse

def traj2dcd(trajpath,output_name):
    if not os.path.exists(trajpath):
        return
    atoms = read(trajpath,index='::1')
    u = mda.Universe(pdbtop, atoms[0][:prot_atom_num].get_positions())    
    with DCDWriter(output_name, n_atoms=u.atoms.n_atoms) as W:
        for idx,frame in enumerate(atoms):
            if idx % stride == 0:
                u.atoms.positions = frame[:prot_atom_num].get_positions()            
                W.write(u)

parser = argparse.ArgumentParser(description="Traj to DCD")
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--pdb', type=str)
parser.add_argument('--num-atoms',type=int)
parser.add_argument('--stride', type=int)

args = parser.parse_args()
trajpath = args.input
output_name = args.output
pdbtop = args.pdb
prot_atom_num = args.num_atoms
stride = args.stride

traj2dcd(trajpath,output_name)
