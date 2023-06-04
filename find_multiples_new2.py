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

def PE(xc, mc, hc, G=4.301e3):
    """ xc - array of positions
        mc - array of masses
        hc - array of smoothing lengths
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, G=G, theta=0.7, method='bruteforce') # G in code units
    return 0.5 * (phic*mc).sum()


def get_orbit(p1, p2, v1, v2, m1, m2, G=4.301e3, h1=0, h2=0):
    """
    Auxiliary function to get binary properties for two particles.

    :param Array-like p1: -- Array with 1st particle position (3D)
    :param Array-like p2: -- Array with 2nd particle position (3D)
    :param Array-like v1: -- Array with 1st particle velocity (3D)
    :param Array-like v2: -- Array with 2nd particle velocity (3D)
    :param float m1: -- 1st particle mass
    :param float m2: -- 2nd particle mass

    :return: The semimajor axis, eccentricity, inclination, the particle separation, com position, com velocity, m1, and m2.
    :rtype: tuple
    """
    dp = np.linalg.norm(p1 - p2)

    com = (m1*p1 + m2*p2)/(m1 + m2)
    com_vel = (m1*v1 + m2*v2)/(m1 + m2)
    ##Particle velocities in com frame
    p1_com = p1 - com
    p2_com = p2 - com
    v1_com = v1 - com_vel
    v2_com = v2 - com_vel

    v12 = (v1_com[0]**2. + v1_com[1]**2. + v1_com[2]**2.)
    v22 = (v2_com[0]**2. + v2_com[1]**2. + v2_com[2]**2.)

    ##Kinetic and potential energies
    ke = 0.5*m1*v12 + 0.5*m2*v22
    ##Potential energy ##TRY REPLACING WITH FUNCTIONALITY FROM PYTREEGRAV...
    # pe = G*m1*m2/dp
    pe = -PE(np.array([p2_com, p1_com]), np.array([m1, m2]), np.array([h1, h2]))

    a_bin = G*(m1*m2)/(2.*(pe-ke))
    ##Angular momentum in binary com
    j_bin = m1*np.cross(p1_com, v1_com) + m2*np.cross(p2_com, v2_com)
    ##Angular momentum of binary com
    j_com = (m1 + m2)*np.cross(com, com_vel)

    #Inclination
    i_bin = np.arccos(np.dot(j_bin, j_com)/np.linalg.norm(j_bin)/np.linalg.norm(j_com))*180./np.pi
    mu = m1*m2/(m1+m2)
    ##Eccentricity of the binary *squared*
    e_bin = np.sqrt(1.-np.linalg.norm(j_bin)**2./(G*(m1+m2)*a_bin)/(mu**2.))
    return a_bin, e_bin, i_bin, dp, com[0], com[1], com[2], com_vel[0], com_vel[1], com_vel[2], m1, m2


def select_in_subregion(x, Ngrid1D=1):
    """
    Partitions array of 3D positions into subregions

    :param Array-like x: Array of positions
    :param int Ngrid1D: Linear dimension of the grid.
    """
    xmin = np.min(x[:, 0])
    xmax = np.max(x[:, 0])
    dx = (xmax-xmin)/(Ngrid1D)

    ymin = np.min(x[:, 1])
    ymax = np.max(x[:, 1])
    dy = (ymax-ymin)/(Ngrid1D)

    zmin = np.min(x[:, 2])
    zmax = np.max(x[:, 2])
    dz = (zmax-zmin)/(Ngrid1D)

    regions = []
    for grid_ind in range(Ngrid1D*Ngrid1D*Ngrid1D):
        x_ind = (grid_ind % Ngrid1D)
        y_ind = ((grid_ind-x_ind) % (Ngrid1D*Ngrid1D))/Ngrid1D
        z_ind = (grid_ind-x_ind-y_ind*Ngrid1D)/(Ngrid1D*Ngrid1D)
        xlim = xmin+x_ind*dx
        ylim = ymin+y_ind*dy
        zlim = zmin+z_ind*dz

        regions.append((x[:, 0] >= xlim) & (x[:, 0] <= (xlim+dx)) & (x[:, 1] >= ylim) & (x[:, 1] <= (ylim+dy)) & (x[:, 2] >= zlim) & (x[:, 2] <= (zlim+dz)))
    return regions

# def check_tides(pos, mass, accel, soft, idx1, idx2, G):
#     """
#     Check whether tidal force is greater than two-body force between two stars.
#
#     :param Array-like mass: Particle positions
#     :param Array-like pos: Particle positions
#     :param Array-like accel: Particle accelerations
#     :param Array-like soft: Softening length
#     :param int idx1: First particle index
#     :param int idx2: Second particle index
#
#     :return: Boolean indicating whether tidal force exceed the internal two-body force.
#     :rtype: bool
#
#     """
#     f2body_i = mass[idx1] * pytreegrav.AccelTarget(np.atleast_2d(pos[idx1]), np.atleast_2d(pos[idx2]),
#                                                    np.atleast_1d(mass[idx2]), softening_target=np.atleast_1d(soft[idx1]),
#                                                    softening_source=np.atleast_1d(soft[idx2]), G=G)
#     com_accel = (mass[idx1] * accel[idx1] + mass[idx2] * accel[idx2]) / (mass[idx1] + mass[idx2])
#     f_tides = mass[idx1] * (accel[idx1] - com_accel) - f2body_i
#     a_tides = f_tides / mass[idx1]
#
#     tidal_crit = (np.linalg.norm(f_tides) < np.linalg.norm(f2body_i))
#     return tidal_crit, a_tides

def check_tides_sys(sys1, sys2, G):
    """
    Check whether tidal force is greater than two-body force between two stars.

    :param System sys1: Particle positions
    :param System sys2: Particle positions
    :param int idx1: First particle index
    :param int idx2: Second particle index

    :return: Boolean indicating whether tidal force exceed the internal two-body force.
    :rtype: bool

    """
    ##Can we avoid these two separate cases?? Dont want to handle single multiplicty stars as a special case.
    ##In any case we still have to modify how we store sub positions.
    sys1_pos = sys1.sub_pos
    sys1_mass = sys1.sub_mass
    sys1_soft = sys1.sub_soft
    ##System 2...
    sys2_pos = sys2.sub_pos
    sys2_mass = sys2.sub_mass
    sys2_soft = sys2.sub_soft
    ##Acceleration of system 1 particles due to system 2 particles
    ##Acceleration here??
    a_internal = pytreegrav.AccelTarget(np.atleast_2d(sys1_pos), np.atleast_2d(sys2_pos),
                                                   np.atleast_1d(sys2_mass), softening_target=np.atleast_1d(sys1_soft),
                                                   softening_source=np.atleast_1d(sys2_soft), G=G)
    ##Acceleration of com of system 1 due to system 2.
    a_internal_com = np.dot(sys1_mass, a_internal) / sys1.mass
    ##Acceleration of com of whole system
    com_accel = (sys1.mass * sys1.accel + sys2.mass * sys2.accel) / (sys1.mass + sys2.mass)
    ##Difference acceleration of system and com acceleration of com ##How do we want to order the subtraction?
    a_tides = (sys1.accel - com_accel) - a_internal_com
    ##Tidal criterion
    tidal_crit = (np.linalg.norm(a_tides) < 2. * np.linalg.norm(a_internal_com))
    ##Check if tides are actually destructive
    compress = np.dot(a_tides, sys2.pos - sys1.pos) > 0

    return (tidal_crit or compress), a_tides

def flatten_ids(id):
    """
    Convert arbitrary nested list of numbers into a 1D numpy array (i.e. flatten hierarchy to get list of ids
    in a given system
    """
    return np.array(str(id).replace("[", "").replace("]", "").split(",")).astype(int)

class system(object):
    """
    Class storing single bound system within the cluster

    :param Array-like p1: Particle position
    :param Array-like v1: Particle velocity
    :param float m1: Particle mass
    :param float h1: Softening length
    :param Array-like id1: Particle ID(s) [Can store hierarchy!]
    :param float accel: Particle acceleration
    :param int sysID: ID that can be used to tag a system

    """
    def __init__(self, p1, v1, m1, h1, id1, accel, sysID, pos_to_spos=False):
        self.pos = np.copy(p1)
        self.vel = np.copy(v1)
        self.mass = m1
        self.soft = h1
        self.accel = accel

        self.orbits = np.zeros((0, 16))
        self.sub_pos = np.zeros((0, 3))
        self.sub_vel = np.zeros((0, 3))
        self.sub_accel = np.zeros((0, 3))
        self.sub_mass = np.zeros(0)
        self.sub_soft = np.zeros(0)
        if pos_to_spos:
            self.sub_pos = np.atleast_2d(self.pos)
            self.sub_vel = np.atleast_2d(self.vel)
            self.sub_mass = np.atleast_1d(self.mass)
            self.sub_soft = np.atleast_1d(self.soft)
            self.sub_accel = np.atleast_2d(self.accel)
        self.sysID = sysID
        self.hierarchy = id1
        self.ids = flatten_ids(id1)

        ##To store tidal accelerations
        self.a_tides = []

    @property
    def multiplicity(self):
        """
        Multiplcity of system
        """
        return len(self.ids)

    def add_orbit(self, orb):
        """
        Add orbit to system
        """
        self.orbits = np.concatenate((self.orbits, orb))

    def add_sub_pos(self, pos):
        """
        Add position of system subcomponent
        """
        self.sub_pos = np.concatenate((self.sub_pos, pos))

    def add_sub_vel(self, vel):
        """
        Add velocity of system subcomponent
        """
        self.sub_vel = np.concatenate((self.sub_vel, vel))

    def add_sub_accel(self, accel):
        """
        Add velocity of system subcomponent
        """
        self.sub_accel = np.concatenate((self.sub_accel, accel))


    def add_sub_mass(self, mass):
        """
        Add mass of system subcomponent
        """
        self.sub_mass = np.concatenate((self.sub_mass, mass))

    def add_sub_soft(self, h):
        """
        Add softening length of system subcomponent
        """
        self.sub_soft = np.concatenate((self.sub_soft, h))

    def add_a_tides(self, a_tides):
        """
        Add tidal acceleration to the system
        """
        self.a_tides = np.concatenate((self.a_tides, a_tides))


class cluster(object):
    """
    Store system information from stellar data in  starforge snapshot, grouping stars hierarchically
    into bound pairs.

    :param Array-like ps: Particle positions
    :param Array-like vs: Particle velocities
    :param Array-like ms: Particle masses
    :param Array-like partsink: Softening lengths
    :param Array-like ids: Particle ids
    :param Array-like accels: Particle accelerations

    """
    def __init__(self, ps, vs, ms, partsink, ids, accels, tides=True, Ngrid1D=1, sma_order=False, mult_max=4):
        self.G = 4.301e3
        self.mult_max = mult_max
        self.Ngrid1D = Ngrid1D
        self.sma_order = sma_order
        self.systems = []
        ##Adding each star as a system
        for ii in range(len(ps)):
            self.systems.append(system(ps[ii], vs[ii], ms[ii], partsink[ii], ids[ii], accels[ii], ii, pos_to_spos=True))
        self.systems = np.array(self.systems)
        self.tides = tides
        ##Partition stars into different subregions -- copied from one of existing binary-finding codes.
        ##Can help with performance.
        self.regions = select_in_subregion(self.get_system_position, Ngrid1D=self.Ngrid1D)
        self.orb_all = []
        self._calculate_orbits()
        conv = False
        ##Look for the most bound systems until we converge.
        while not conv:
            systems_start = [ss.multiplicity for ss in self.systems]
            self._find_binaries_all()
            systems_end = [ss.multiplicity for ss in self.systems]
            conv = (systems_start == systems_end)
        # print("test")

    @property
    def get_system_position(self):
        return np.array([ss.pos for ss in self.systems])

    @property
    def get_system_vel(self):
        return np.array([ss.vel for ss in self.systems])

    @property
    def get_system_mass(self):
        return np.array([ss.mass for ss in self.systems])

    @property
    def get_system_ids(self):
        return [ss.ids for ss in self.systems]

    @property
    def get_system_ids_b(self):
        return np.array([ss.sysID for ss in self.systems])

    @property
    def get_system_hierarchies(self):
        return [ss.hierarchy for ss in self.systems]

    @property
    def get_system_soft(self):
        return np.array([ss.soft for ss in self.systems])

    @property
    def get_system_accel(self):
        return np.array([ss.accel for ss in self.systems])

    @property
    def get_system_mult(self):
        return np.array([ss.multiplicity for ss in self.systems])

    def _calculate_orbits(self):
        """
        Computes pairwise orbits in each subregion, populating orb_all.
        Each entry in orb_all contains 16 entries:

        The semimajor axis, eccentricity, inclination, the particle separation, com position, com velocity, m1, m2, and the system IDs.

        orb_all is then adjusted as systems are combined hierarchically, using the methods
        orb_adjust_delete and orb_adjust_add.
        """
        for region in self.regions:
            pos = self.get_system_position[region]
            vel = self.get_system_vel[region]
            mass = self.get_system_mass[region]
            soft = self.get_system_soft[region]
            idx = np.array(range(len(self.systems)))[region]
            orb_region = []
            combos_all = np.array(list(combinations(list(range(len(pos))), 2)))
            for jj, combo in enumerate(combos_all):
                i = combo[0]
                j = combo[1]
                orb_region.append(np.concatenate((get_orbit(pos[i], pos[j], vel[i], vel[j], mass[i], mass[j], G=self.G, h1=soft[0], h2=soft[1]),
                                                  [self.systems[idx[i]].ids[0], self.systems[idx[j]].ids[0], self.systems[idx[i]].sysID, self.systems[idx[j]].sysID])))
            self.orb_all.append(np.array(orb_region))

    def _find_binaries_all(self):
        """
        Find most-bound binaries in each subregion
        """
        for ii in range(len(self.regions)):
            self._find_bin_region(ii)

    def _find_bin_region(self, ii):
        """
        Find the binary with the largest binding energy in a given subregion.

        :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------param int ii: index of the subregion.
        """
        orb_all = self.orb_all[ii]
        if len(orb_all) < 1:
            return
        sysIDs = self.get_system_ids_b
        pos = self.get_system_position
        mass = self.get_system_mass
        soft = self.get_system_soft
        accel = self.get_system_accel

        ens = -self.G*orb_all[:, 10]*orb_all[:, 11]/(2.*orb_all[:, 0])
        if self.sma_order:
            ens = orb_all[:, 0]
        en_order = np.argsort(ens)
        orb_all = orb_all[en_order]
        ##Filter out negative smas to save time here
        orb_all = orb_all[orb_all[:, 0] > 0]

        for row in orb_all:
            ID1 = int(row[-2])
            ID2 = int(row[-1])
            idx1 = np.where(sysIDs == ID1)[0][0]
            idx2 = np.where(sysIDs == ID2)[0][0]

            mult_total = self.systems[idx1].multiplicity + self.systems[idx2].multiplicity
            ###Tidal criterion:
            #tidal_crit, at0 = check_tides(pos, mass, accel, soft, idx1, idx2, self.G)
            tidal_crit_1, at1 = check_tides_sys(self.systems[idx1], self.systems[idx2], self.G)
            tidal_crit_2, at2 = check_tides_sys(self.systems[idx2], self.systems[idx1], self.G)
            tidal_crit = tidal_crit_1 and tidal_crit_2
            ##Symmetrize tidal criterion? (e.g. Call again with arguments flipped)
            tidal_crit = (tidal_crit) or (not self.tides)
            ##Check that binary is bound, multiplicity is less than four, and that the binary is tidally stable. Tides can be turned off by setting self.tides to False.
            if row[0] > 0 and (mult_total <= self.mult_max) and tidal_crit:
                print("adding {0}".format(mult_total))
                ID_NEW = self._combine_binaries(row)
                ##Put add operation first to deal with special case of only three stars
                self._orbit_adjust_add(ii, ID_NEW)
                self._orbit_adjust_delete(ii, ID1, ID2)
                ##Store tidal acceleration (proper setter)
                self.systems[-1].add_a_tides(at1)
                self.systems[-1].add_a_tides(at2)

                return

    def _combine_binaries(self, row):
        """
        Adjusts array of systems after a binary has been found.
        The members of the bound pair are removed from the systems array and
        then a single new system replaces them.

        :param row Array-like: Binary data of bound pair from orb_all.
        :return: ID for new system that is added
        :rtype: int
        """

        sysIDs = self.get_system_ids_b
        sysID_max = np.max(sysIDs)

        filt = ~np.isin(sysIDs, row[-2:].astype(int).ravel())
        systems_new = self.systems[filt]
        hierarchies = self.get_system_hierarchies

        idx1 = np.where(sysIDs == row[-2])[0][0]
        idx2 = np.where(sysIDs == row[-1])[0][0]

        m1 = self.systems[idx1].mass
        m2 = self.systems[idx2].mass
        h1 = self.systems[idx1].soft
        h2 = self.systems[idx2].soft
        a_com = (m1 * self.systems[idx1].accel + m2 * self.systems[idx2].accel) / (m1 + m2)

        ss_new = system(row[4:7], row[7:10], row[10] + row[11], h1 + h2, [hierarchies[idx1], hierarchies[idx2]], a_com,
                        sysID_max + 1)
        ss_new.add_orbit(self.systems[idx1].orbits)
        ss_new.add_orbit(self.systems[idx2].orbits)
        ss_new.add_orbit([row])

        ss_new.add_sub_pos(self.systems[idx1].sub_pos)
        ss_new.add_sub_pos(self.systems[idx2].sub_pos)
        # ss_new.add_sub_pos([self.systems[idx1].pos])
        # ss_new.add_sub_pos([self.systems[idx2].pos])

        ss_new.add_sub_vel(self.systems[idx1].sub_vel)
        ss_new.add_sub_vel(self.systems[idx2].sub_vel)
        # ss_new.add_sub_vel([self.systems[idx1].vel])
        # ss_new.add_sub_vel([self.systems[idx2].vel])

        ss_new.add_sub_accel(self.systems[idx1].sub_accel)
        ss_new.add_sub_accel(self.systems[idx2].sub_accel)

        ss_new.add_sub_soft(self.systems[idx1].sub_soft)
        ss_new.add_sub_soft(self.systems[idx2].sub_soft)
        # ss_new.add_sub_soft([self.systems[idx1].soft])
        # ss_new.add_sub_soft([self.systems[idx2].soft])

        ss_new.add_sub_mass(self.systems[idx1].sub_mass)
        ss_new.add_sub_mass(self.systems[idx2].sub_mass)
        # ss_new.add_sub_mass([self.systems[idx1].mass])
        # ss_new.add_sub_mass([self.systems[idx2].mass])

        systems_new = np.concatenate((systems_new, [ss_new]))

        self.systems = systems_new
        return sysID_max + 1

    def _orbit_adjust_delete(self, ii, ID1, ID2):
        """
        Used to adjust orb_all after a binary has been found.

        Once we find the most bound binary stars, they are combined into a single system.
        The entries corresponding to these stars in orb_all should then be deleted.
        """
        combos_all = self.orb_all[ii][:, -2:].astype(int)
        test1 = np.where(np.any(ID1 == combos_all, axis=1))[0]
        test2 = np.where(np.any(ID2 == combos_all, axis=1))[0]

        to_delete = np.concatenate((test1, test2))
        self.orb_all[ii] = np.delete(self.orb_all[ii], to_delete, axis=0)

    def _orbit_adjust_add(self, ii, ID_NEW):
        """
        Used to adjust orb_all after a binary has been found.

        Once we find the most bound binary stars, they are combined into a single system.
        The entries corresponding to these stars in orb_all should then be deleted.
        """

        regionIDs = np.unique(self.orb_all[ii][:, -2].astype(int).ravel())
        sysIDs = self.get_system_ids_b
        regionIDs = regionIDs[np.isin(regionIDs, sysIDs)]
        pos = self.get_system_position
        vel = self.get_system_vel
        mass = self.get_system_mass
        ids = self.get_system_ids
        soft = self.get_system_soft

        idx1 = np.where(sysIDs == ID_NEW)[0][0]
        for id_it in regionIDs:
            j = np.where(id_it == sysIDs)[0][0]
            tmp = get_orbit(pos[idx1], pos[j], vel[idx1], vel[j], mass[idx1], mass[j], G=self.G, h1=soft[idx1], h2=soft[j])
            tmp = np.concatenate((tmp, [ids[idx1][0], ids[j][0], ID_NEW, id_it]))
            self.orb_all[ii] = np.append(self.orb_all[ii], tmp)
            self.orb_all[ii].shape = (-1, 16)


def main():
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Name of snapshot to read")
    parser.add_argument("--sma_order", action="store_true", help="Assemble hierarchy by sma instead of binding energy")
    parser.add_argument("--halo_mass_file", default="", help="Name of file containing gas halo mass around sink particles")
    parser.add_argument("--mult_max", type=int, default=4, help="Multiplicity cut (4).")
    parser.add_argument("--ngrid", type=int, default=1, help="Number of subgrids to use. Higher number will be faster,"
                                                             " but less accurate (1)")
    args = parser.parse_args()

    snapshot_file = args.snap
    sma_order = args.sma_order
    # den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, tcgs, unit_base = load_data(snapshot_file, res_limit=1e-3)
    # cl = cluster(partpos, partvels, partmasses, partids)
    den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base = load_data(snapshot_file, res_limit=1e-3)
    ##TO DO: UPDATE ONCE WE HAVE METHOD TO STORE HALO DATA.
    halo_masses = np.zeros(len(partmasses))
    if args.halo_mass_file:
        halo_masses = np.genfromtxt(args.halo_mass_file)[:, 0]
    partmasses += halo_masses

    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    partpos = partpos.astype(np.float64)
    partmasses = partmasses.astype(np.float64)
    partsink = partsink.astype(np.float64)

    ##TO DO: MAKE SURE THIS IS CONSISTENT WITH THE SIMULATION (Theta, tree gravity versus brute force)
    accel_gas = pytreegrav.AccelTarget(partpos, xuniq, muniq, softening_target=partsink, softening_source=huniq, theta=0.5, G=4.301e3)
    accel_stars = pytreegrav.Accel(partpos, partmasses, partsink, theta=0.5, G=4.301e3)
    #
    cl = cluster(partpos, partvels, partmasses, partsink, partids, accel_stars + accel_gas,
                 sma_order=sma_order, mult_max=args.mult_max, Ngrid1D=args.ngrid)
    with open("testing_" + snapshot_file.replace(".hdf5", "")+"_TidesTrue" +
              "_smaOrder{0}_mult{1}_ngrid{2}_hm{3}".format(sma_order, args.mult_max, args.ngrid, len(args.halo_mass_file) > 0) + ".p", "wb") as ff:
        pickle.dump(cl, ff)

    cl = cluster(partpos, partvels, partmasses, partsink, partids, accel_stars + accel_gas, tides=False,
                 sma_order=sma_order, mult_max=args.mult_max, Ngrid1D=args.ngrid)
    with open("testing_" + snapshot_file.replace(".hdf5", "")+"_TidesFalse" +
              "_smaOrder{0}_mult{1}_ngrid{2}_hm{3}".format(sma_order, args.mult_max, args.ngrid, len(args.halo_mass_file) > 0) +".p", "wb") as ff:
        pickle.dump(cl, ff)



if __name__ == "__main__":
    main()