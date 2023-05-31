import pickle
import numpy as np
import sys

sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import find_multiples_new2
from find_multiples_new2 import cluster, system
import pytreegrav
import progressbar
import argparse
import h5py

GN = 4.301e3

def PE(xc, mc, hc, G=GN):
    """ xc - array of positions
        mc - array of masses
        hc - array of smoothing lengths
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, G=G, theta=0.7, method='bruteforce')  # G in code units
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



def check_tides_gen(pos, mass, accel, soft, idx1, idx2, G, compress=False, tides_factor=1):
    """
    Estimate of tidal force for SINGLE gas cell + multiple...position, mass, etc. of gas cell
    has to come first in pos, mass, etc.

    :param Array-like mass: Particle positions
    :param Array-like pos: Particle positions
    :param Array-like accel: Particle accelerations
    :param Array-like soft: Softening length
    :param int idx1: First particle index
    :param int idx2: Second particle index
    :param float G: Gravitational constant
    :param bool compress (False): Filtering out compressive tides
    :param float tides_factor (1): Prefactor used in comparison of tidal and internal forces.

    :return: Boolean indicating whether tidal force exceed the internal two-body force.
    :rtype: bool

    """
    f2body_i = mass[idx1] * pytreegrav.AccelTarget(np.atleast_2d(pos[idx1]), np.atleast_2d(pos[idx2]),
                                               np.atleast_1d(mass[idx2]),
                                               softening_target=np.atleast_1d(soft[idx1]),
                                               softening_source=np.atleast_1d(soft[idx2]), G=G)
    com_accel = (mass[idx1] * accel[idx1] + np.sum(mass[idx2]) * accel[idx2]) / (mass[idx1] + np.sum(mass[idx2]))
    ##Safeguard for rounding error??
    f_tides = mass[idx1] * (accel[idx1] - com_accel) - f2body_i

    tidal_crit = (np.linalg.norm(f_tides) < tides_factor * np.linalg.norm(f2body_i))
    ##Add compressive criterion HERE!!
    com_pos = np.average(pos[idx2], weights=mass[idx2], axis=0)
    if compress:
        compress_check = np.dot(f_tides, com_pos - pos[idx1]) > 0
        tidal_crit = tidal_crit or compress_check


    return tidal_crit

def blob_setup(sys1):
    """
    Bookkeeping function for get_gas_mass_bound
    """
    cumul_masses = np.array([sys1.mass])
    cumul_soft = np.array([sys1.soft])

    cumul_pos = np.copy([sys1.pos])
    cumul_vel = np.copy([sys1.vel])
    cumul_accel = np.copy(sys1.accel)

    cumul_u = np.zeros(len(cumul_pos))

    com_masses = sys1.mass
    com_pos = np.copy(sys1.pos)
    com_vel = np.copy(sys1.vel)

    blob = {'cumul_masses': cumul_masses,
            'cumul_pos': cumul_pos,
            'cumul_soft': cumul_soft,
            'cumul_vel': cumul_vel,
            'cumul_u': cumul_u,
            'cumul_accel': cumul_accel,
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
    blob['cumul_accel'] = (blob['com_masses'] * blob['cumul_accel'] + muniq1[idx] * accel_gas1[idx]) /\
                          (muniq1[idx] + blob['com_masses'])
    blob['cumul_masses'] = np.append(blob['cumul_masses'], muniq1[idx])
    blob['cumul_u'] = np.append(blob['cumul_u'], uuniq1[idx])
    blob['cumul_pos'] = np.vstack([blob['cumul_pos'], xuniq1[idx]])
    blob['cumul_vel'] = np.vstack([blob['cumul_vel'], vuniq1[idx]])
    blob['cumul_soft'] = np.append(blob['cumul_soft'], huniq1[idx])
    blob['com_masses'] = np.sum(blob['cumul_masses'])
    blob['com_pos'] = np.average(blob['cumul_pos'], weights=blob['cumul_masses'], axis=0)
    blob['com_vel'] = np.average(blob['cumul_vel'], weights=blob['cumul_masses'], axis=0)

    return blob


# def get_bound_bins(partpos, bound_index, in1=0, out1=0.51, delta1=0.01):
#     absc = np.arange(in1, out1, delta1)
#     partpos = np.copy(partpos)
#     nsinks = len(partpos)
#
#
#     for ii in range(nsinks):
#         ##Halo data for one sink
#         halo_dat = np.genfromtxt(path1 + "_{0}".format(ii))
#         ##Distance from gas particles to this sink
#         ##Not good practice since we are using xuniq from the outer scope!!
#         d = xuniq - partpos[ii]
#         d = np.sum(d * d, axis=1) ** .5
#         ##Bin gas particles by distance and store.
#         val, bins = np.histogram(d[d < 0.5], bins=absc)
#         val2, bins = np.histogram(d[halo_dat.astype(int)], bins=absc)
#         gas_all1[ii] = val
#         gas_bound1[ii] = val2
#
#     return bins, gas_all1, gas_bound1

def get_gas_mass_bound_refactor(sys1, gas_data, sinkpos, G=GN, cutoff=0.1, non_pair=False, compress=False, tides_factor=1):
    """
    Get to gas mass bound to a system. This is meant to be applied to a *single star.*

    :param System sys1: System we are interested
    :param tuple gas_data: All the positions, velocities, masses, softening lengths, and accelerations of all gas
    :param Array-like sinkpos: Position of all sinks
    :param float G: Gravitational constant
    :param float cutoff: Distance up to which we look for bound gas
    :param bool non_pair: Flag to include non-pairwise interactions.
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
    for idx in progressbar.progressbar(ord1):
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
        tmp_pos = [xuniq1[idx], blob['cumul_pos']]
        tmp_mass = [muniq1[idx], blob['cumul_masses']]
        tmp_soft = [huniq1[idx], blob['cumul_soft']]
        tmp_accel = [accel_gas1[idx], blob['cumul_accel']]
        tide_crit = check_tides_gen(tmp_pos, tmp_mass, tmp_accel, tmp_soft, 0, 1, G, compress=compress,
                                    tides_factor=tides_factor)

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

#Add cumulative argument for non-pairwise algorithm!!! Add vuniq to arguments as well
# def get_gas_mass_bound_og(sys1, xuniq, muniq, huniq, accel_gas, G=GN, cutoff=0.1, non_pair=False):
#     if sys1.multiplicity > 1:
#         cumul_masses = np.copy(sys1.sub_mass)
#         cumul_pos = np.copy(sys1.sub_pos)
#         cumul_soft = np.copy(sys1.sub_soft)
#         cumul_vel = np.copy(sys1.sub_vel)
#     else:
#         cumul_masses = np.array([sys1.mass])
#         cumul_pos = np.copy([sys1.pos])
#         cumul_soft = np.array([sys1.soft])
#         cumul_vel = np.copy([sys1.vel])
#     cumul_u = np.zeros(len(cumul_pos))
#     cumul_accel = sys1.accel
#     com_masses = sys1.mass
#     com_pos = sys1.pos
#     com_vel = sys1.vel
#     # vuniq_mag = np.sum((vuniq - sys1.vel) * (vuniq - sys1.vel), axis=1) ** .5
#
#     d = xuniq - com_pos
#     d = np.sum(d * d, axis=1)
#     ord1 = np.argsort(d)
#     # d_sorted = d[ord1]**.5
#     d_max = 0
#
#     halo_mass = 0.
#     for idx in progressbar.progressbar(ord1):
#         if d[idx]**.5 > cutoff:
#             break
#         ##Likely unnecessary since no change in results/performance.
#         # if denuniq[idx] < 0.247:
#         #     continue
#         ##Use velocity relative to the cumulative center-of-mass
#         tmp_vrel = np.linalg.norm(vuniq[idx] - com_vel)
#         if tmp_vrel > (2. * G * (com_masses + muniq[idx]) / d[idx]**.5)**.5:
#             continue
#
#         ##Maybe can also replace with PotentialTarget?! Gives a lot of extra information...
#         # pe0 = muniq[idx] * pytreegrav.Potential(np.vstack((cumul_pos, xuniq[idx])),
#         #                                         np.append(cumul_masses, muniq[idx]),
#         #                                         softening=np.append(cumul_soft, huniq[idx]),
#         #                                         G=G, theta=0.7, method='bruteforce')
#
#         pe1 = muniq[idx] * pytreegrav.PotentialTarget(np.atleast_2d(xuniq[idx]), cumul_pos, cumul_masses,
#                                                 softening_target=np.atleast_1d(huniq[idx]), softening_source=cumul_soft,
#                                                 G=G, theta=0.7, method='bruteforce')[-1]
#         ke1 = KE(np.vstack([com_pos, xuniq[idx]]), np.append(com_masses, muniq[idx]),
#                  np.vstack([com_vel, vuniq[idx]]), np.append(0, uuniq[idx]))
#         # ke1 = KE(np.vstack([sys1.pos, xuniq[idx]]), np.append(sys1.mass, muniq[idx]),
#         #          np.vstack([sys1.vel, vuniq[idx]]), np.append(0, uuniq[idx]))
#         tmp_pos = [xuniq[idx], cumul_pos]
#         tmp_mass = [muniq[idx], cumul_masses]
#         tmp_soft = [huniq[idx], cumul_soft]
#         tmp_accel = [accel_gas[idx], cumul_accel]
#         tide_crit = check_tides_gen(tmp_pos, tmp_mass, tmp_accel, tmp_soft, 0, 1, G)
#
#         if (pe1 + ke1 < 0) and (tide_crit):
#             if non_pair:
#
#                 cumul_accel = (np.sum(cumul_masses) * cumul_accel + muniq[idx] * accel_gas[idx]) / (
#                             muniq[idx] + np.sum(cumul_masses))
#                 cumul_masses = np.append(cumul_masses, muniq[idx])
#                 cumul_pos = np.vstack([cumul_pos, xuniq[idx]])
#                 cumul_vel = np.vstack([cumul_vel, vuniq[idx]])
#                 cumul_soft = np.append(cumul_soft, huniq[idx])
#                 cumul_u = np.append(cumul_u, uuniq[idx])
#                 com_masses = np.sum(cumul_masses)
#                 com_pos = np.average(cumul_pos, weights=cumul_masses, axis=0)
#                 com_vel = np.average(cumul_vel, weights=cumul_masses, axis=0)
#
#             d_max = d[idx]**.5
#             halo_mass += muniq[idx]
#
#     return halo_mass, d_max



parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
parser.add_argument("snap", help="Index of snapshot to read")
parser.add_argument("--non_pair", action="store_true", help="Flag to turn on non-pairwise algorithm")
parser.add_argument("--compress", action="store_true", help="Filter out compressive tidal forces")
parser.add_argument("--tides_factor", type=float, default=1.0, help="Prefactor for check of tidal criterion (1)")
parser.add_argument("--cutoff", type=float, default=0.1, help="Outer cutoff to look for bound gas (0.1 pc)")


args = parser.parse_args()

snap_idx = args.snap
cutoff = args.cutoff
non_pair = args.non_pair

with open("snapshot_{0}_TidesFalse_smaOrderFalse_mult2.p".format(snap_idx), "rb") as ff:
    cl = pickle.load(ff)

sys_masses = np.array(cl.get_system_mass)
sys_mult = np.array([ss.multiplicity for ss in cl.systems])
sys_id = np.array(cl.get_system_ids)
systems1 = [ss for ss in cl.systems]

##TO DO: GENERATE THIS DATA IF THIS IS NOT PRESENT
accel_gas = np.genfromtxt('accel_gas_{0}'.format(snap_idx))
accel_stars = np.genfromtxt('accel_stars_{0}'.format(snap_idx))
snap_file = 'snapshot_{0}.hdf5'.format(snap_idx)
den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base =\
find_multiples_new2.load_data(snap_file, res_limit=1e-3)

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

halo_masses_sing = np.zeros(len(partpos))
max_dist_sing = np.zeros(len(partpos))
ids_sing = np.zeros(len(partpos))
gas_dat_h5 = h5py.File("halo_masses_sing_{0}_np{1}_c{2}_comp{3}_tf{4}.hdf5".format(snap_idx, non_pair, cutoff, args.compress,
                                                                           args.tides_factor), 'a')
for ii, pp in enumerate(partpos):
    sys_tmp = find_multiples_new2.system(partpos[ii], partvels[ii], partmasses[ii], partsink[ii], partids[ii], accel_stars[ii], 0)
    gas_data = (xuniq, vuniq, muniq, huniq, uuniq, accel_gas)
    ##TO DO: CALCULATE AND STORE FULL PROFILE HERE...USE LOGARITHMIC GRID FROM SOFTENING LENGTH TO CUTOFF
    halo_masses_sing[ii], max_dist_sing[ii], bound_index, rad_bins, halo_mass_bins = get_gas_mass_bound_refactor(sys_tmp, gas_data, partpos,
                                                                                        G=GN, cutoff=cutoff, non_pair=non_pair,
                                                                                        compress=args.compress, tides_factor=args.tides_factor)
    ids_sing[ii] = partids[ii]
    gas_dat_h5.create_dataset("halo_{0}".format(partids[ii]), data=np.transpose((rad_bins, halo_mass_bins)))

    # output_file = "gas_halo_data/bound_stars_{0}_np{1}_c{2}_comp{3}_tf{4}_{5}".format(snap_idx, non_pair, cutoff, args.compress,
    #                                                                  args.tides_factor, ii)
    # np.savetxt(output_file, bound_index)

gas_dat_h5.close()
output_file ="gas_halo_data/halo_masses_sing_{0}_np{1}_c{2}_comp{3}_tf{4}".format(snap_idx, non_pair, cutoff, args.compress,
                                                                     args.tides_factor)
np.savetxt(output_file, np.transpose((halo_masses_sing, ids_sing, max_dist_sing)))

