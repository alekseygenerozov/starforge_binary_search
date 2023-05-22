import numpy as np
import h5py
from itertools import combinations
import pickle
import pytreegrav
import argparse

import warnings


def load_data(file, res_limit=0.0):
    """ file - h5pdf5 STARFORGE snapshot
        res_limit - minimum mass resolution to include in analyis (in code units)
    """
    # Load snapshot data
    f = h5py.File(file, 'r')

    # Mask to remove any cells with mass below the cell resolution
    # (implemented specifically to remove feedback cells if desired)
    mask = (f['PartType0']['Masses'][:] >= res_limit * 0.999)
    mask3d = np.array([mask, mask, mask]).T

    # Read in gas properties
    # Mass density
    den = f['PartType0']['Density'][:] * mask
    # Spatial positions
    x = f['PartType0']['Coordinates'] * mask3d

    # Mass of each cell/partical
    m = f['PartType0']['Masses'][:] * mask
    # Calculation smoothing length, useful for weighting and/or visualization
    h = f['PartType0']['SmoothingLength'][:] * mask
    # Internal (thermal) energy
    u = f['PartType0']['InternalEnergy'][:] * mask
    v = f['PartType0']['Velocities'] * mask3d
    b = f['PartType0']['MagneticField'][:] * mask3d
    t = f['PartType0']['Temperature'][:] * mask
    # Fraction of molecular material in each cell
    fmol = f['PartType0']['MolecularMassFraction'][:] * mask
    # To get molecular gas density do: den*fmol*fneu*(1-helium_mass_fraction)/(2.0*mh), helium_mass_fraction=0.284
    fneu = f['PartType0']['NeutralHydrogenAbundance'][:] * mask

    if 'PartType5' in f.keys():
        partpos = f['PartType5']['Coordinates'][:]
        partmasses = f['PartType5']['Masses'][:]
        partvels = f['PartType5']['Velocities'][:]
        partids = f['PartType5']['ParticleIDs'][:]
        partsink = (f['PartType5']['SinkRadius'][:])
    else:
        partpos = []
        partmasses = [0]
        partids = []
        partvels = [0, 0, 0]
        partsink = []

    time = f['Header'].attrs['Time']
    unitlen = f['Header'].attrs['UnitLength_In_CGS']
    unitmass = f['Header'].attrs['UnitMass_In_CGS']
    unitvel = f['Header'].attrs['UnitVelocity_In_CGS']
    unitb = 1e4  # f['Header'].attrs['UnitMagneticField_In_CGS'] If not defined

    unit_base = {'UnitLength': unitlen, 'UnitMass': unitmass, 'UnitVel': unitvel, 'UnitB': unitb}

    # Unit base information specifies conversion between code units and CGS
    # Example: To convert to density in units of g/cm^3 do: den*unit_base['UnitMass']/unit_base['UnitLength']**3

    tcgs = time * (unit_base['UnitLength'] / unit_base['UnitVel']) / (3600.0 * 24.0 * 365.0 * 1e6)
    print("Snapshot time in %f Myr" % (tcgs))

    del f
    return den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base

def main():
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Name of snapshot to read")
    args = parser.parse_args()

    den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base = load_data(args.snap)

    nsinks = len(partpos)
    partids.shape = (nsinks, -1)
    partsink.shape = (nsinks, -1)
    partmasses.shape = (nsinks, -1)

    np.savetxt(args.snap.replace(".hdf5", ".sink"), np.hstack((partids, partpos, partvels, partsink, partmasses)))


if __name__ == "__main__":
    main()