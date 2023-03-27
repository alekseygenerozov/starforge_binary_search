import pickle
import numpy as np
import sys
sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import find_multiples_new2
from find_multiples_new2 import cluster, system
import pytreegrav
import progressbar

GN = 4.301e3

def PE(xc, mc, hc, G=4.301e3):
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
    """
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

snap_idx = sys.argv[1]
with open("snapshot_{0}_TidesFalse_smaOrderFalse_mult2.p".format(snap_idx), "rb") as ff:
    cl = pickle.load(ff)

sys_masses = np.array(cl.get_system_mass)
sys_pos = np.array(cl.get_system_position)
sys_vel = np.array(cl.get_system_vel)
sys_soft = np.array(cl.get_system_soft)
sys_mult = np.array([ss.multiplicity for ss in cl.systems])
sys_id = np.array(cl.get_system_ids)
sys_accel = np.array([ss.accel for ss in cl.systems])

sys_sub_masses = [ss.sub_mass for ss in cl.systems]
sys_sub_pos = [ss.sub_pos for ss in cl.systems]
sys_sub_soft = [ss.sub_soft for ss in cl.systems]

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
vuniq = vuniq.astype(np.float64)
xuniq = xuniq.astype(np.float64)
muniq = muniq.astype(np.float64)
huniq = huniq.astype(np.float64)
uuniq = uuniq.astype(np.float64)
partpos = partpos.astype(np.float64)
partmasses = partmasses.astype(np.float64)
partsink = partsink.astype(np.float64)

bin_smas = np.concatenate(np.array([ss.orbits[:, 0] for ss in cl.systems], dtype=object)[sys_mult == 2])

halo_masses_bin = np.zeros(len(sys_pos))
for ii, pp in enumerate(sys_pos):
    if sys_mult[ii] != 2:
        continue
    d = xuniq - sys_pos[ii]
    d = np.sum(d*d, axis=1)
    ord1 = np.argsort(d)
    for idx in progressbar.progressbar(ord1):
        if d[idx]**.5 > 0.1:
            break

        pe1 = muniq[idx] * pytreegrav.Potential(np.vstack((sys_sub_pos[ii], xuniq[idx])),\
                     np.append(sys_sub_masses[ii], muniq[idx]),\
                     np.append(sys_sub_soft[ii], huniq[idx]), G=GN, theta=0.7, method='bruteforce')[-1]
        ke1 = KE(np.array([sys_pos[ii], xuniq[idx]]), np.array([sys_masses[ii], muniq[idx]]),\
                 np.array([sys_vel[ii], vuniq[idx]]), np.array([0, uuniq[idx]]))
        tmp_pos = [xuniq[idx], sys_sub_pos[ii]]
        tmp_mass = [muniq[idx], sys_sub_masses[ii]]
        tmp_soft = [huniq[idx], sys_sub_soft[ii]]
        tmp_accel = [accel_gas[idx], sys_accel[ii]]
        tide_crit = check_tides_gen(tmp_pos, tmp_mass, tmp_accel, tmp_soft, 0, 1, GN)

        if (pe1 + ke1 < 0) and (tide_crit):
            halo_masses_bin[ii] += muniq[idx]

halo_masses_sing = np.zeros(len(partpos))
for ii, pp in enumerate(partpos):
    d = xuniq - partpos[ii]
    d = np.sum(d*d, axis=1)
    ord1 = np.argsort(d)
    for idx in progressbar.progressbar(ord1):
        if d[idx]**.5 > 0.1:
            break
        sma = get_sma_opt(partpos[ii].astype(np.float64), xuniq[idx].astype(np.float64),\
                          partvels[ii].astype(np.float64), vuniq[idx].astype(np.float64),\
                          partmasses[ii], muniq[idx], partsink[ii], huniq[idx], 0., uuniq[idx], GN)
        tmp_pos = [xuniq[idx], partpos[ii]]
        tmp_mass = [muniq[idx], partmasses[ii]]
        tmp_soft = [huniq[idx], partsink[ii]]
        tmp_accel = [accel_gas[idx], accel_stars[ii]]
        tide_crit = check_tides_gen(tmp_pos, tmp_mass, tmp_accel, tmp_soft, 0, 1, GN)

        if (sma > 0) and (tide_crit):
            halo_masses_sing[ii] += muniq[idx]

bin_ids_non_zero = sys_id[(sys_mult == 2) & (halo_masses_bin > 0)]
bin_masses_non_zero = sys_masses[(sys_mult == 2) & (halo_masses_bin > 0)]
halo_masses_bin_non_zero = halo_masses_bin[(sys_mult == 2) & (halo_masses_bin > 0)]
bin_smas_non_zero = bin_smas[halo_masses_bin[sys_mult == 2] > 0]

compare_masses_list = []
for ii, row in enumerate(bin_ids_non_zero):
    idx1 = np.where(partids == row[0])[0][0]
    idx2 = np.where(partids == row[1])[0][0]
    compare_masses_list.append((halo_masses_bin_non_zero[ii], halo_masses_sing[idx1], halo_masses_sing[idx2], bin_smas_non_zero[ii], idx1, idx2, bin_masses_non_zero[ii]))

np.savetxt("compare2_masses_list_{0}".format(sys.argv[1]), compare_masses_list)

