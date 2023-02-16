#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2022 Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import argparse
import os
import sys
from ase.io import read


import numpy as np

from sgdml import __version__
from sgdml.utils import io, ui

if sys.version[0] == '3':
    raw_input = input


parser = argparse.ArgumentParser(
    description='Creates a dataset from any input format supported by ASE.'
)
parser.add_argument(
    'dataset',
    metavar='<dataset>',
    type=argparse.FileType('r'),
    help='path to input dataset file',
)
parser.add_argument(
    '-o',
    '--overwrite',
    dest='overwrite',
    action='store_true',
    help='overwrite existing dataset file',
)
args = parser.parse_args()
dataset = args.dataset


name = os.path.splitext(os.path.basename(dataset.name))[0]
dataset_file_name = name + '.npz'

dataset_exists = os.path.isfile(dataset_file_name)
if dataset_exists and args.overwrite:
    print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing dataset file.')
if not dataset_exists or args.overwrite:
    print('Writing dataset to \'{}\'...'.format(dataset_file_name))
else:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
        + ' Dataset \'{}\' already exists.'.format(dataset_file_name)
    )

mols = read(dataset.name, index=':')

# filter incomplete outputs from trajectory
mols = [mol for mol in mols if mol.get_calculator() is not None]

lattice, R, z, E, F = None, None, None, None, None

calc = mols[0].get_calculator()

print("\rNumber geometries: {:,}".format(len(mols)))
#print("\rAvailable properties: " + ', '.join(calc.results))
print()

if 'forces' not in calc.results:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
        + ' Forces are missing in the input file!'
    )

lattice = np.array(mols[0].get_cell().T)
if not np.any(lattice):
    print(
        ui.color_str('[INFO]', bold=True)
        + ' No lattice vectors specified.'
    )
    lattice = None

Z = np.array([mol.get_atomic_numbers() for mol in mols])
all_z_the_same = (Z == Z[0]).all()
if not all_z_the_same:
    sys.exit(
        ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
        + ' Order of atoms changes accross dataset.'
    )

R = np.array([mol.get_positions() for mol in mols])
z = Z[0]

if 'Energy' in mols[0].info:
    E = np.array([float(mol.info['Energy']) for mol in mols])
else:
    E = np.array([mol.get_potential_energy() for mol in mols])
F = np.array([mol.get_forces() for mol in mols])

print('Please provide a name for this dataset. Otherwise the original filename will be reused.')
custom_name = raw_input('> ').strip()
if custom_name != '':
    name = custom_name

print('Please provide a descriptor for the level of theory used to create this dataset.')
theory = raw_input('> ').strip()
if theory == '':
    theory = 'unknown'

# Base variables contained in every model file.
base_vars = {
    'type': 'd',
    'code_version': __version__,
    'name': name,
    'theory': theory,
    'R': R,
    'z': z,
    'F': F,
}

base_vars['F_min'], base_vars['F_max'] = np.min(F.ravel()), np.max(F.ravel())
base_vars['F_mean'], base_vars['F_var'] = np.mean(F.ravel()), np.var(F.ravel())

print('If you want to convert your original length unit, please provide a conversion factor (default: 1.0): ')
R_to_new_unit = raw_input('> ').strip()
if R_to_new_unit != '':
    R_to_new_unit = float(R_to_new_unit)
else:
    R_to_new_unit = 1.0

print('If you want to convert your original energy unit, please provide a conversion factor (default: 1.0): ')
E_to_new_unit = raw_input('> ').strip()
if E_to_new_unit != '':
    E_to_new_unit = float(E_to_new_unit)
else:
    E_to_new_unit = 1.0


r_unit = raw_input('> ').strip()
if r_unit != '':
    base_vars['r_unit'] = r_unit


e_unit = raw_input('> ').strip()
if e_unit != '':
    base_vars['e_unit'] = e_unit

if E is not None:
    base_vars['E'] = E * E_to_new_unit
    base_vars['E_min'], base_vars['E_max'] = np.min(E), np.max(E)
    base_vars['E_mean'], base_vars['E_var'] = np.mean(E), np.var(E)
else:
    print(ui.color_str('[INFO]', bold=True) + ' No energy labels found in dataset.')

base_vars['R'] *= R_to_new_unit
base_vars['F'] *= E_to_new_unit / R_to_new_unit

if lattice is not None:
    base_vars['lattice'] = lattice

base_vars['md5'] = io.dataset_md5(base_vars)
np.savez_compressed(dataset_file_name, **base_vars)
print(ui.color_str('[DONE]', fore_color=ui.GREEN, bold=True))
