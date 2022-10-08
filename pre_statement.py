import ase
from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
from dscribe.descriptors import SOAP
import numpy as np
import torch
from ase.io import read

file_path = "vasprun.xml"
atom_file_path = "file_lib/atomtype_info_list.npy"
pos_file_path = "file_lib/position_list.npy"
forces_file_path = "file_lib/track_poscar_info_list.npy"
energies_file_path = "file_lib/track_energy_list.npy"

atom_type = np.load(atom_file_path)
pos = np.load(pos_file_path)
forces = np.load(forces_file_path)
energies = np.load(energies_file_path)

soap = SOAP(
        species=["Si"],
        periodic=False,
        rcut=5.0,
        sigma=0.5,
        nmax=3,
        lmax=0,
)
step = len(pos)
traj = []
energies = []
forces = []
r = np.linspace(0, 1000, len(pos))

for i in range(len(pos)):
    a = read(file_path, index= i)
    traj.append(a)
    energies.append(a.get_total_energy())
    forces.append(a.get_forces())
# Plot the energies to validate them
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
line, = ax.plot(r, energies)
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")
plt.show()

derivatives, descriptors = soap.derivatives(
        traj,
        method="analytical"
)

# Save to disk for later training
np.save("npy_lib/r.npy", r)
np.save("npy_lib/E.npy", energies)
np.save("npy_lib/D.npy", descriptors)
np.save("npy_lib/dD_dr.npy", derivatives)
np.save("npy_lib/F.npy", forces)


