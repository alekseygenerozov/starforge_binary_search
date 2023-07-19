import pickle
import numpy as np
import sys
##Code uses functionality in find_multiples_new2
# sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import find_multiples_new2
from find_multiples_new2 import cluster, system
import pytreegrav
# import progressbar
import argparse
import h5py
import multiprocessing
import functools

def PE(xc, mc, hc, G=4.301e3):
    """ xc - array of positions
        mc - array of masses
        hc - array of smoothing lengths
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, G=G, theta=0.5, method='bruteforce')  # G in code units
    return 0.5 * (phic * mc).sum()


# Calculate kinetic energy of a set of cells, include internal energy
def KE(xc, mc, vc, uc):
    """ xc - array of positions
        mc - array of masses
        vc - array of velocities
        uc - array of internal energies
    """
    ## velocity w.r.t. com velocity
    v_bulk = np.average(vc, weights=mc, axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well ** 2, axis=1)
    return (mc * (vSqr / 2 + uc)).sum()



# def check_tides_gen(pos, mass, accel, soft, idx1, idx2, G, compress=False, tides_factor=8.0):
#     """
#     Estimate of tidal force for SINGLE gas cell + multiple...position, mass, etc. of gas cell
#     has to come first in pos, mass, etc.
#
#     :param Array-like mass: Particle positions
#     :param Array-like pos: Particle positions
#     :param Array-like accel: Particle accelerations
#     :param Array-like soft: Softening length
#     :param int idx1: First particle index
#     :param int idx2: Second particle index
#     :param float G: Gravitational constant
#     :param bool compress (False): Filtering out compressive tides
#     :param float tides_factor (2): Prefactor used in comparison of tidal and internal forces.
#
#     :return: Boolean indicating whether tidal force exceed the internal two-body force.
#     :rtype: bool
#
#     """
#     ##THIS IS NOT QUITE RIGHT SINCE THE ACCELERATIONS SHOULD BE COMPUTED USING TREE METHOD...
#     f2body_i = mass[idx1] * pytreegrav.AccelTarget(np.atleast_2d(pos[idx1]), np.atleast_2d(pos[idx2]),
#                                                np.atleast_1d(mass[idx2]),
#                                                softening_target=np.atleast_1d(soft[idx1]),
#                                                softening_source=np.atleast_1d(soft[idx2]), G=G, method='bruteforce')
#     com_accel = (mass[idx1] * accel[idx1] + np.sum(mass[idx2]) * accel[idx2]) / (mass[idx1] + np.sum(mass[idx2]))
#     ##Safeguard for rounding error??
#     f_tides = mass[idx1] * (accel[idx1] - com_accel) - f2body_i
#
#     ##Set tides factor automatically based on the inclination of the binary?? Check literature...
#     tidal_crit = (np.linalg.norm(f_tides) < tides_factor * np.linalg.norm(f2body_i))
#     com_pos = np.average(pos[idx2], weights=mass[idx2], axis=0)
#     ##Compressive tides...
#     if compress:
#         compress_check = np.dot(f_tides, com_pos - pos[idx1]) > 0
#         tidal_crit = tidal_crit or compress_check
#
#     return tidal_crit, f_tides/mass[idx1]

def blob_setup(sys1):
    """
    Bookkeeping function for get_gas_mass_bound
    """
    cumul_masses = np.array([sys1.mass])
    cumul_soft = np.array([sys1.soft])

    cumul_pos = np.copy([sys1.pos])
    cumul_vel = np.copy([sys1.vel])
    cumul_u = np.zeros(len(cumul_pos))

    com_accel = np.copy(sys1.accel)
    com_masses = sys1.mass
    com_pos = np.copy(sys1.pos)
    com_vel = np.copy(sys1.vel)

    blob = {'cumul_masses': cumul_masses,
            'cumul_pos': cumul_pos,
            'cumul_soft': cumul_soft,
            'cumul_vel': cumul_vel,
            'cumul_u': cumul_u,
            'com_accel': com_accel,
            'com_masses': com_masses,
            'com_pos': com_pos,
            'com_vel': com_vel}

    return blob

def add_to_blob(blob, gas_data, idx):
    """
    Bookkeeping function for get_gas_mass_bound
    """
    ##Could make copies but may take up too much memory...
    xuniq1, vuniq1, muniq1, huniq1, uuniq1, accel_gas1 = gas_data

    ##This is really the com_accel: Rename to keep the pattern
    blob['cumul_masses'] = np.append(blob['cumul_masses'], muniq1[idx])
    blob['cumul_u'] = np.append(blob['cumul_u'], uuniq1[idx])
    blob['cumul_pos'] = np.vstack([blob['cumul_pos'], xuniq1[idx]])
    blob['cumul_vel'] = np.vstack([blob['cumul_vel'], vuniq1[idx]])
    blob['cumul_soft'] = np.append(blob['cumul_soft'], huniq1[idx])
    blob['com_accel'] = (blob['com_masses'] * blob['com_accel'] + muniq1[idx] * accel_gas1[idx]) /\
                          (muniq1[idx] + blob['com_masses'])
    blob['com_masses'] = np.sum(blob['cumul_masses'])
    blob['com_pos'] = np.average(blob['cumul_pos'], weights=blob['cumul_masses'], axis=0)
    blob['com_vel'] = np.average(blob['cumul_vel'], weights=blob['cumul_masses'], axis=0)

    return blob

def get_gas_mass_bound_refactor(sys1, gas_data, sinkpos, G=4.301e3, cutoff=0.5, non_pair=False, compress=False, tides_factor=8):
    """
    Get to gas mass bound to a system. This is meant to be applied to a *single star.*

    :param System sys1: System we are interested
    :param tuple gas_data: All the positions, velocities, masses, softening lengths, and accelerations of all gas
    :param Array-like sinkpos: Position of all sinks
    :param float G: Gravitational constant
    :param float cutoff: Distance up to which we look for bound gas
    :param bool non_pair: Flag to include non-pairwise interactions.
    :param bool compress: Whether to filter out compressive tidal forces (False).
    :param float tides_factor: Prefactor to use in comparison of tidal criterion.

    """
    blob = blob_setup(sys1)
    xuniq1, vuniq1, muniq1, huniq1, uuniq1, accel_gas1 = gas_data

    d = xuniq1 - blob['com_pos']
    d = np.sum(d * d, axis=1)**.5
    ord1 = np.argsort(d)
    d_max = 0
    halo_mass = 0.
    rad_bins = np.geomspace(sys1.soft, cutoff, 100)
    halo_mass_bins = np.zeros(len(rad_bins))
    bound_index = []
    particle_indices = range(len(xuniq1))
    for idx in ord1:
        if d[idx] > cutoff:
            break
        dall = (xuniq1[idx] - sinkpos)
        dall = np.sum(dall * dall, axis=1)**.5
        if not np.isclose(np.min(dall), d[idx]):
            continue

        ##Use velocity relative to the cumulative center-of-mass
        tmp_vrel = np.linalg.norm(vuniq1[idx] - blob['com_vel'])
        ##Performance but unexpected decreases bound gas
        if tmp_vrel > np.sqrt((2. * G * (blob['com_masses'] + muniq1[idx])) / d[idx]):
            continue

        pe1 = muniq1[idx] * pytreegrav.PotentialTarget(np.atleast_2d(xuniq1[idx]), blob['cumul_pos'],
                                                blob['cumul_masses'],
                                                softening_target=np.atleast_1d(huniq1[idx]),
                                                softening_source=blob['cumul_soft'],
                                                G=G, method='bruteforce')[-1]
        ke1 = KE(np.vstack([blob['com_pos'], xuniq1[idx]]), np.append(blob['com_masses'], muniq1[idx]),
                 np.vstack([blob['com_vel'], vuniq1[idx]]), np.append(0, uuniq1[idx]))

        ##Could refactor this part
        tmp_sys1 = find_multiples_new2.system(xuniq1[idx], vuniq1[idx], muniq1[idx],
                                              huniq1[idx], 0, accel_gas1[idx], 0, pos_to_spos=True)
        tmp_sys2 = find_multiples_new2.system(blob['cumul_pos'], blob['cumul_vel'], blob['cumul_masses'],
                                                blob['cumul_soft'], 0, blob['com_accel'], 0, pos_to_spos=True)
        tide_crit, at1 = find_multiples_new2.check_tides_sys(tmp_sys1, tmp_sys2, G=G, compress=compress, tides_factor=tides_factor)

        if (pe1 + ke1 < 0) and (tide_crit):
            if non_pair:
                blob = add_to_blob(blob, gas_data, idx)

            d_max = d[idx]
            halo_mass += muniq1[idx]
            ##Storing binned data
            halo_mass_idx = np.searchsorted(rad_bins, d[idx])
            halo_mass_bins[halo_mass_idx] += muniq1[idx]
            bound_index.append(particle_indices[idx])

    halo_mass_bins = np.cumsum(halo_mass_bins)
    return halo_mass, d_max, bound_index, .5 * (rad_bins[:-1] + rad_bins[1:]), halo_mass_bins[1:]




def get_mass_bound_manager(part_data, gas_data, ii, **kwargs):
    partpos, partvels, partmasses, partsink, partids, accel_stars = part_data
    sys_tmp = find_multiples_new2.system(partpos[ii], partvels[ii], partmasses[ii], partsink[ii], partids[ii],
                                         accel_stars[ii], 0)
    res = get_gas_mass_bound_refactor(sys_tmp, gas_data, partpos, **kwargs)
    halo_mass, max_dist, bound_index, rad_bins, halo_mass_bins = res

    return halo_mass, max_dist, np.transpose((rad_bins, halo_mass_bins))

def main():
    ##Fix GN from the simulation data rather than hard-coding...
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Index of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    parser.add_argument("--non_pair", action="store_true", help="Flag to turn on non-pairwise algorithm")
    parser.add_argument("--compress", action="store_true", help="Filter out compressive tidal forces")
    parser.add_argument("--tides_factor", type=float, default=8.0, help="Prefactor for check of tidal criterion (8.0)")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Outer cutoff to look for bound gas (0.5 pc)")

    args = parser.parse_args()

    snap_idx = args.snap
    cutoff = args.cutoff
    non_pair = args.non_pair

    snap_file = args.snap_base + '_{0}.hdf5'.format(snap_idx)

    den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base =\
    find_multiples_new2.load_data(snap_file, res_limit=1e-3)
    GN = 6.672e-8 * (unit_base['UnitVel'] ** 2. * unit_base['UnitLength'] / (unit_base['UnitMass'])) ** -1.

    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    vuniq = v[indx]
    uuniq = u[indx]
    denuniq = den[indx]
    vuniq = vuniq.astype(np.float64)
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    uuniq = uuniq.astype(np.float64)
    denuniq = denuniq.astype(np.float64)
    partpos = partpos.astype(np.float64)
    partmasses = partmasses.astype(np.float64)
    partsink = partsink.astype(np.float64)
    ##Calculate the accelerations of the sink particles due to gas and stars. Use the full collection of stars...
    # accel_gas = pytreegrav.AccelTarget(partpos, xuniq, muniq, softening_target=partsink, softening_source=huniq, theta=0.5, G=4.301e3)
    # accel_stars = pytreegrav.Accel(partpos, partmasses, partsink, theta=0.5, G=4.301e3)
    ##TO DO: USE TREE FOR GAS CELL ACCELERATIONS. FOR SINKS: TREE OR DIRECT SUM? NOT SURE?
    ##Combined positions for computing accelerations
    pos_all = np.vstack((xuniq, partpos))
    mass_all = np.concatenate((muniq, partmasses))
    soft_all = np.concatenate((huniq, partsink))
    tree1 = pytreegrav.ConstructTree(pos_all, mass_all, softening=soft_all)
    ##Acceleration of gas due to stars
    accel_gas = pytreegrav.AccelTarget(xuniq, None, None,
                    softening_target=huniq, softening_source=soft_all,
                    tree=tree1, theta=0.5, G=GN)
    ##Acceleration of stars/sinks. Accelerations due to gas are computed with tree. Acceleration due to sinks are
    ##computed with direct summation...Not sure about this
    accel_stars_gas = pytreegrav.AccelTarget(partpos, xuniq, muniq, softening_target=partsink, softening_source=huniq,
                                             theta=0.5, G=GN, method="tree")
    accel_stars_stars = pytreegrav.Accel(partpos, partmasses, partsink, G=GN, method="bruteforce")
    accel_stars = accel_stars_gas + accel_stars_stars
    # accel_gas = np.genfromtxt("accel_gas_{0}".format(snap_idx))
    # accel_stars = np.genfromtxt("accel_stars_{0}".format(snap_idx))

    halo_masses_sing = np.zeros(len(partpos))
    max_dist_sing = np.zeros(len(partpos))
    gas_dat_h5 = h5py.File("halo_masses_sing_{0}_np{1}_c{2}_comp{3}_tf{4}.hdf5".format(snap_idx, non_pair, cutoff, args.compress,
                                                                               args.tides_factor), 'a')
    gas_data = (xuniq, vuniq, muniq, huniq, uuniq, accel_gas)
    part_data = (partpos, partvels, partmasses, partsink, partids, accel_stars)
    f_to_iter = functools.partial(get_mass_bound_manager, part_data, gas_data,
                      G=GN, cutoff=cutoff, non_pair=non_pair, compress=args.compress, tides_factor=args.tides_factor)
    with multiprocessing.Pool(NTASKS) as pool:
        for ii, halo_dat_full in enumerate(pool.map(f_to_iter, range(len(halo_masses_sing)))):
            halo_masses_sing[ii], max_dist_sing[ii], halo_dat = halo_dat_full
            gas_dat_h5.create_dataset("halo_{0}".format(partids[ii]), data=halo_dat)

    gas_dat_h5.close()
    output_file ="halo_masses_sing_{0}_np{1}_c{2}_comp{3}_tf{4}".format(snap_idx, non_pair, cutoff, args.compress,
                                                                         args.tides_factor)
    np.savetxt(output_file, np.transpose((halo_masses_sing, partids, max_dist_sing)))


if __name__ == "__main__":
    main()
