import pickle
import numpy as np
import sys

sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import find_multiples_new2
from find_multiples_new2 import cluster, system
import pytreegrav
import progressbar
import argparse

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


# @njit('float64(float64[:], float64[:], float64[:], float64[:], float64, float64, float64)')
def get_sma_opt(p1, p2, v1, v2, m1, m2, h1, h2, u1, u2, G):
    """        # ke1 = KE(np.vstack([com_pos, xuniq[idx]]), np.append(com_masses, muniq[idx]),\
        #          np.vstack([com_vel, vuniq[idx]]), np.append(0, uuniq[idx]))
    Auxiliary function to get binary properties for two particles.

    :param Array-like p1: -- Array with 1st particle position (3D)
    :param Array-like p2: -- Array with 2nd particle position (3D)
    :param Array-like v1: -- Array with 1st particle velocity (3D)
    :param Array-like v2: -- Array with 2nd particle velocity (3D)
    :param float m1: -- 1st particle mass
    :param float m2: -- 2nd particle mass
    :param float h1: -- 1st particle softening length
    :param float h2: -- 2nd particle softening length
    :param float u1: -- 1st particle internal energy
    :param float u2: -- 2nd particle internal energy

    :return: The semimajor axis, eccentricity, inclination, the particle separation, com position, com velocity, m1, and m2.
    :rtype: tuple
    """
    dp = np.linalg.norm(p1 - p2)

    com = (m1 * p1 + m2 * p2) / (m1 + m2)
    com_vel = (m1 * v1 + m2 * v2) / (m1 + m2)
    ##Particle velocities in com frame
    p1_com = p1 - com
    p2_com = p2 - com
    v1_com = v1 - com_vel
    v2_com = v2 - com_vel

    v12 = (v1_com[0] ** 2. + v1_com[1] ** 2. + v1_com[2] ** 2.)
    v22 = (v2_com[0] ** 2. + v2_com[1] ** 2. + v2_com[2] ** 2.)

    ##Kinetic and potential energies (In principle for gas should also include internal energy)
    #     ke = 0.5*m1*v12 + 0.5*m2*v22
    ke = KE(np.array([p1_com, p2_com]), np.array([m1, m2]), np.array([v1_com, v2_com]), np.array([u1, u2]))
    ##Potential energy;
    # pe = G*m1*m2/dp
    pe = -PE(np.array([p1_com, p2_com]), np.array([m1, m2]), np.array([h1, h2]), G=G)
    a_bin = G * (m1 * m2) / (2. * (pe - ke))

    return a_bin


# Estimate of tidal force for SINGLE gas cell + multiple...position, mass, etc. of gas cell
# has to come first in pos, mass, etc.
def check_tides_gen(pos, mass, accel, soft, idx1, idx2, G):
    """
    Check whether tidal force is greater than two-body force between two stars.

    :param Array-like mass: Particle positions
    :param Array-like pos: Particle positions
    :param Array-like accel: Particle accelerations
    :param Array-like soft: Softening length
    :param int idx1: First particle index
    :param int idx2: Second particle index

    :return: Boolean indicating whether tidal force exceed the internal two-body force.
    :rtype: bool

    """
    f2body_i = mass[idx1] * pytreegrav.AccelTarget(np.atleast_2d(pos[idx1]), np.atleast_2d(pos[idx2]),
                                               np.atleast_1d(mass[idx2]),
                                               softening_target=np.atleast_1d(soft[idx1]),
                                               softening_source=np.atleast_1d(soft[idx2]), G=G)
    com_accel = (mass[idx1] * accel[idx1] + np.sum(mass[idx2]) * accel[idx2]) / (mass[idx1] + np.sum(mass[idx2]))
    f_tides = mass[idx1] * (accel[idx1] - com_accel) - f2body_i

    tidal_crit = (np.linalg.norm(f_tides) < np.linalg.norm(f2body_i))
    return tidal_crit

##Add cumulative argument for non-pairwise algorithm!!!
def get_gas_mass_bound(sys1, xuniq, muniq, huniq, accel_gas, G=GN, cutoff=0.1, non_pair=False):
    if sys1.multiplicity > 1:
        cumul_masses = np.copy(sys1.sub_mass)
        cumul_pos = np.copy(sys1.sub_pos)
        cumul_soft = np.copy(sys1.sub_soft)
        cumul_vel = np.copy(sys1.sub_vel)
    else:
        cumul_masses = np.array([sys1.mass])
        cumul_pos = np.copy([sys1.pos])
        cumul_soft = np.array([sys1.soft])
        cumul_vel = np.copy([sys1.vel])
    cumul_u = np.zeros(len(cumul_pos))
    cumul_accel = sys1.accel
    com_masses = sys1.mass
    com_pos = sys1.pos
    com_vel = sys1.vel
    # vuniq_mag = np.sum((vuniq - sys1.vel) * (vuniq - sys1.vel), axis=1) ** .5

    d = xuniq - com_pos
    d = np.sum(d * d, axis=1)
    ord1 = np.argsort(d)
    # d_sorted = d[ord1]**.5
    d_max = 0

    halo_mass = 0.
    for idx in progressbar.progressbar(ord1):
        if d[idx]**.5 > cutoff:
            break
        ##Likely unnecessary since no change in results/performance.
        if denuniq[idx] < 0.247:
            continue
        ##Use velocity relative to the cumulative center-of-mass
        tmp_vrel = np.linalg.norm(vuniq[idx] - com_vel)
        if tmp_vrel > (2. * G * np.sum(cumul_masses) / d[idx]**.5)**.5:
            continue

        ##Maybe can also replace with PotentialTarget?! Gives a lot of extra information...
        # pe1 = muniq[idx] * pytreegrav.Potential(np.vstack((cumul_pos, xuniq[idx])),
        #                                         np.append(cumul_masses, muniq[idx]),
        #                                         np.append(cumul_soft, huniq[idx]),
        #                                         G=G, theta=0.7, method='bruteforce')[-1]

        pe1 = muniq[idx] * pytreegrav.PotentialTarget(np.atleast_2d(xuniq[idx]), cumul_pos, cumul_masses,
                                                softening_target=np.atleast_1d(huniq[idx]), softening_source=cumul_soft,
                                                G=G, theta=0.7, method='bruteforce')[-1]
        ke1 = KE(np.vstack([com_pos, xuniq[idx]]), np.append(com_masses, muniq[idx]),
                 np.vstack([com_vel, vuniq[idx]]), np.append(0, uuniq[idx]))
        # ke1 = KE(np.vstack([sys1.pos, xuniq[idx]]), np.append(sys1.mass, muniq[idx]),
        #          np.vstack([sys1.vel, vuniq[idx]]), np.append(0, uuniq[idx]))
        tmp_pos = [xuniq[idx], cumul_pos]
        tmp_mass = [muniq[idx], cumul_masses]
        tmp_soft = [huniq[idx], cumul_soft]
        tmp_accel = [accel_gas[idx], cumul_accel]
        tide_crit = check_tides_gen(tmp_pos, tmp_mass, tmp_accel, tmp_soft, 0, 1, G)

        if (pe1 + ke1 < 0) and (tide_crit):
            if non_pair:

                cumul_accel = (np.sum(cumul_masses) * cumul_accel + muniq[idx] * accel_gas[idx]) / (
                            muniq[idx] + np.sum(cumul_masses))
                cumul_masses = np.append(cumul_masses, muniq[idx])
                cumul_pos = np.vstack([cumul_pos, xuniq[idx]])
                cumul_vel = np.vstack([cumul_vel, vuniq[idx]])
                cumul_soft = np.append(cumul_soft, huniq[idx])
                cumul_u = np.append(cumul_u, uuniq[idx])
                com_masses = np.sum(cumul_masses)
                com_pos = np.average(cumul_pos, weights=cumul_masses, axis=0)
                com_vel = np.average(cumul_vel, weights=cumul_masses, axis=0)

            d_max = d[idx]**.5
            halo_mass += muniq[idx]

    return halo_mass, d_max


parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
parser.add_argument("snap", help="Index of snapshot to read")
parser.add_argument("--non_pair", action="store_true", help="Flag to turn on non-pairwise algorithm")
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

# bin_smas = np.concatenate(np.array([ss.orbits[:, 0] for ss in cl.systems], dtype=object)[sys_mult == 2])

# halo_masses_bin = np.zeros(len(systems1))
# max_dist_bin = np.zeros(len(systems1))
# ids1 = np.zeros(len(halo_masses_bin))
# ids2 = np.zeros(len(halo_masses_bin))
# for ii, pp in enumerate(systems1):
#     if sys_mult[ii] != 2:
#         continue
#     halo_masses_bin[ii], max_dist_bin[ii] = get_gas_mass_bound(systems1[ii], xuniq, muniq, huniq, accel_gas, G=GN, cutoff=cutoff,
#                                              non_pair=non_pair)
#     ids1[ii], ids2[ii] = systems1[ii].ids
# #
# np.savetxt("halo_masses_bin_{0}_np{1}_c{2}".format(snap_idx, non_pair, cutoff),
#            np.transpose((halo_masses_bin, ids1, ids2, max_dist_bin)))

halo_masses_sing = np.zeros(len(partpos))
max_dist_sing = np.zeros(len(partpos))
ids_sing = np.zeros(len(partpos))
for ii, pp in enumerate(partpos):
    sys_tmp = find_multiples_new2.system(partpos[ii], partvels[ii], partmasses[ii], partsink[ii], partids[ii], accel_stars[ii], 0)
    halo_masses_sing[ii], max_dist_sing[ii] = get_gas_mass_bound(sys_tmp, xuniq, muniq, huniq, accel_gas, G=GN, cutoff=cutoff, non_pair=non_pair)
    ids_sing[ii] = partids[ii]

np.savetxt("tmp_halo_masses_sing_{0}_np{1}_c{2}".format(snap_idx, non_pair, cutoff),
           np.transpose((halo_masses_sing, ids_sing, max_dist_sing)))

