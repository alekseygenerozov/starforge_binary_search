import numpy as np
import h5py
from itertools import combinations
import pickle
import pytreegrav
import argparse

import warnings
import sys

# from obsolete.rebound_runs.hayden.end_incDistro_test import times

sys.path.append("/home/aleksey/rebound")
sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
import rebound
from find_multiples_new2 import cluster
import starforge_constants as sfc

def test_tides():
    sma_planet = 1.0
    m_star = 1.0
    m_planet = 3.0e-6
    m_moon = 3.0e-10

    ibin = 0.0
    fbin = 0.01
    ebin = 1e-3
    fts = np.geomspace(0.62, 0.68)

    for ii in range(len(fts)):
        # delta = deltas[ii]
        abin =  1.0 *  ((m_planet) / (m_star) / 3.) ** (1. / 3.) * sma_planet #* delta

        sim = rebound.Simulation()
        sim.G = sfc.GN
        sim.add(m=m_star, x=1, y=1, z=1)  # Star
        sim.add(m=m_planet, a=sma_planet, inc=1e-4, e=1e-4)  # Planet
        ##Try different phases...
        sim.add(m=m_moon, a=abin, e=ebin, inc=ibin, primary=sim.particles[1], f=fbin)
        sim.integrate(1e-20)

        partpos = np.array([pp.xyz for pp in sim.particles])
        partvels = np.array([pp.vxyz for pp in sim.particles])
        partmasses = np.array([pp.m for pp in sim.particles])
        accel_stars = np.array([[pp.ax, pp.ay, pp.az] for pp in sim.particles])
        partsink = np.zeros(len(sim.particles))
        partids = np.array([1, 2, 3])

        cl = cluster(partpos, partvels, partmasses, partsink, partids, accel_stars, sma_order=True,
                     tides_factor=fts[ii], tides=True, compress=False)
        print(ii, fts[ii], cl.systems[0].hierarchy)
        # if ii == 0:
        #     assert str(cl.systems[0].hierarchy) == '[[2, 3], 1]'
        # else:
        #     assert str(cl.systems[0].hierarchy) != '[[2, 3], 1]'

def main():
    test_tides()

if __name__ == "__main__":
    main()