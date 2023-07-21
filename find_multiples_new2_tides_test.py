import numpy as np
import h5py
from itertools import combinations
import pickle
import pytreegrav
import argparse

import warnings
import sys
sys.path.append("/home/aleksey/rebound")
import rebound
from find_multiples_new2 import cluster

def main():
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("--compress", action="store_true", help="Consider binaries with compressive tides as stable")
    parser.add_argument("--tides_factor", type=float, default=8.0, help="Prefactor for check of tidal criterion (8.0)")

    args = parser.parse_args()
    sma_planet = 1.0
    m_star = 1.0
    m_planet = 3.0e-6
    m_moon = 3.0e-10
    GN = 4.301e3

    inc_grid = np.linspace(0.01, np.pi - 0.1, 20)
    f_grid = np.linspace(0, 2. * np.pi, 20)
    deltas = np.geomspace(0.5, 2.5, 20)

    DELTAS, INCS, FS = np.meshgrid(deltas, inc_grid, f_grid, indexing='ij')
    INCS = INCS.ravel()
    FS = FS.ravel()
    DELTAS = DELTAS.ravel()
    ebin = 1e-3

    for ii in range(len(INCS)):
        ibin = INCS[ii]
        fbin = FS[ii]
        delta = DELTAS[ii]
        abin = ((m_planet) / (m_star)) ** (1. / 3.) * sma_planet * delta

        sim = rebound.Simulation()
        sim.G = GN
        sim.add(m=m_star, x=1, y=1, z=1)  # Star
        sim.add(m=m_planet, a=sma_planet, inc=1e-4, e=1e-4)  # Planet
        sim.add(m=m_moon, a=abin, e=ebin, inc=ibin, primary=sim.particles[1], f=fbin)
        sim.integrate(1e-20)

        partpos = np.array([pp.xyz for pp in sim.particles])
        partvels = np.array([pp.vxyz for pp in sim.particles])
        partmasses = np.array([pp.m for pp in sim.particles])
        accel_stars = np.array([[pp.ax, pp.ay, pp.az] for pp in sim.particles])
        partsink = np.zeros(len(sim.particles))
        partids = np.array([1, 2, 3])

        cl = cluster(partpos, partvels, partmasses, partsink, partids, accel_stars, sma_order=True,
                     tides_factor=args.tides_factor, tides=True, compress=args.compress)
        with open("tide_test{0}.p".format(ii), "wb") as ff:
            pickle.dump(cl, ff)
        with open("moon_pos{0}".format(ii), "w") as ff:
            diff = np.array(sim.particles[2].xyz) - np.array(sim.particles[1].xyz)
            ff.write("{0} {1} {2}\n".format(diff[0], diff[1], diff[2]))




if __name__ == "__main__":
    main()