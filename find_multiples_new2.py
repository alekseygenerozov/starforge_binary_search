import numpy as np
import h5py
from itertools import combinations
import pickle
import pytreegrav


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


def get_orbit(p1, p2, v1, v2, m1, m2, G=4.301e3):
    """
    Auxiliary function to get binary properties for two particles.

    p1, p2 -- particle position
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
    ##Potential energy; Assumes G = 1
    pe = G*m1*m2/dp

    a_bin = G*(m1*m2)/(2.*(pe-ke))
    ##Angular momentum in binary com
    j_bin = m1*np.cross(p1_com, v1_com) + m2*np.cross(p2_com, v2_com)
    ##Angular momentum of binary com
    j_com = (m1 + m2)*np.cross(p1_com, v1_com)

    #Inclination
    i_bin = np.arccos(np.dot(j_bin, j_com)/np.linalg.norm(j_bin)/np.linalg.norm(j_com))*180./np.pi
    mu = m1*m2/(m1+m2)
    ##Eccentricity of the binary *squared*
    e_bin = (1.-np.linalg.norm(j_bin)**2./(G*(m1+m2)*a_bin)/(mu**2.))
    return a_bin, e_bin, i_bin, dp, com[0], com[1], com[2], com_vel[0], com_vel[1], com_vel[2], m1, m2


def select_in_subregion(x, max_dist=0.1):
    dx=np.max(x[:,0])-np.min(x[:,0]);dy=np.max(x[:,1])-np.min(x[:,1]);dz=np.max(x[:,2])-np.min(x[:,2]);
    #Ngrid1D = int(np.clip(np.min([dx, dy, dz]) / max_dist, 1, np.max([5, len(x) ** 0.33])));
    Ngrid1D = 1

    xmin=np.min(x[:,0]);xmax=np.max(x[:,0]); dx=(xmax-xmin)/(Ngrid1D)
    ymin=np.min(x[:,1]);ymax=np.max(x[:,1]); dy=(ymax-ymin)/(Ngrid1D)
    zmin=np.min(x[:,2]);zmax=np.max(x[:,2]); dz=(zmax-zmin)/(Ngrid1D)

    regions = []
    for grid_ind in range(Ngrid1D*Ngrid1D*Ngrid1D):
        x_ind = ( grid_ind % Ngrid1D);
        y_ind = ( (grid_ind-x_ind) % (Ngrid1D*Ngrid1D))/Ngrid1D;
        z_ind = (grid_ind-x_ind-y_ind*Ngrid1D)/(Ngrid1D*Ngrid1D);
        xlim = xmin+x_ind*dx; ylim = ymin+y_ind*dy; zlim = zmin+z_ind*dz;
        regions.append((x[:,0]>=xlim) & (x[:,0]<=(xlim+dx)) & (x[:,1]>=ylim) & (x[:,1]<=(ylim+dy)) & (x[:,2]>=zlim) & (x[:,2]<=(zlim+dz)))
    return regions



class system(object):
    def __init__(self, p1, v1, m1, h1, id1, accel, sysID):
        self.pos = np.copy(p1)
        self.vel = np.copy(v1)
        self.mass = m1
        self.soft = h1
        self.accel = accel

        self.orbits = np.zeros((0, 14))
        self.ids = np.atleast_1d(id1)
        self.sysID = sysID
        # self.child_pos
        # self.child_vel

    @property
    def multiplicity(self):
        return len(self.ids)

    def add_orbit(self, orb):
        self.orbits = np.concatenate((self.orbits, orb))


class cluster(object):
    def __init__(self, ps, vs, ms, partsink, ids, accels, tides=True):
        self.G = 4.301e3
        self.max_dist = 0.1
        self.systems = []
        ##Adding each star as a system
        for ii in range(len(ps)):
            self.systems.append(system(ps[ii], vs[ii], ms[ii], partsink[ii], ids[ii], accels[ii], ii))
        self.systems = np.array(self.systems)
        self.tides = tides
        ##Compute orbits of stars in different subregions -- select in subregion copied from previous code.
        self.regions = select_in_subregion(self.get_system_position, max_dist = self.max_dist)
        self.orb_all = []
        self._calculate_orbits()
        conv = False
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
    def get_system_soft(self):
        return np.array([ss.soft for ss in self.systems])

    @property
    def get_system_accel(self):
        return np.array([ss.accel for ss in self.systems])

    def _find_binaries_all(self):
        for ii in range(len(self.regions)):
            self._find_bin_region(ii)

    def _calculate_orbits(self):
        for region in self.regions:
            pos = self.get_system_position[region]
            vel = self.get_system_vel[region]
            mass = self.get_system_mass[region]
            idx = np.array(range(len(self.systems)))[region]
            orb_region = []
            combos_all = np.array(list(combinations(list(range(len(pos))), 2)))
            for jj, combo in enumerate(combos_all):
                i = combo[0]
                j = combo[1]
                orb_region.append(np.concatenate((get_orbit(pos[i], pos[j], vel[i], vel[j], mass[i], mass[j], G=self.G),
                                                  [self.systems[idx[i]].sysID, self.systems[idx[j]].sysID])))
            self.orb_all.append(np.array(orb_region))
        self.orb_all = self.orb_all


    def _orbit_adjust_delete(self, ii, ID1, ID2):
        ##Have to account for the fact that binary deletion just took place in the indexing -- Otherwise things will go wrong!
        combos_all = self.orb_all[ii][:, -2:].astype(int)
        test1 = np.where(np.any(ID1 == combos_all, axis=1))[0]
        test2 = np.where(np.any(ID2 == combos_all, axis=1))[0]

        to_delete = np.concatenate((test1, test2))
        self.orb_all[ii] = np.delete(self.orb_all[ii], to_delete, axis=0)

    def _orbit_adjust_add(self, ii, ID_NEW):
        regionIDs = np.unique(self.orb_all[ii][:, -2].astype(int).ravel())
        sysIDs = self.get_system_ids_b
        pos = self.get_system_position
        vel = self.get_system_vel
        mass = self.get_system_mass

        idx1 = np.where(sysIDs == ID_NEW)[0][0]
        for id_it in regionIDs:
            j = np.where(id_it == sysIDs)[0][0]
            tmp = get_orbit(pos[idx1], pos[j], vel[idx1], vel[j], mass[idx1], mass[j], G=self.G)
            tmp = np.concatenate((tmp, [ID_NEW, id_it]))
            self.orb_all[ii] = np.append(self.orb_all[ii], tmp)
            self.orb_all[ii].shape = (-1, 14)

    def _find_bin_region(self, ii):
        orb_all = self.orb_all[ii]
        if len(orb_all) < 1:
            return
        sysIDs = self.get_system_ids_b
        pos = self.get_system_position
        mass = self.get_system_mass
        soft = self.get_system_soft
        accel = self.get_system_accel

        ens = -self.G*orb_all[:, 10]*orb_all[:, 11]/(2.*orb_all[:, 0])
        en_order = np.argsort(ens)
        orb_all = orb_all[en_order]

        for row in orb_all:
            ID1 = int(row[-2])
            ID2 = int(row[-1])
            idx1 = np.where(sysIDs == ID1)[0][0]
            idx2 = np.where(sysIDs == ID2)[0][0]

            mult_total = self.systems[idx1].multiplicity + self.systems[idx2].multiplicity
            ###Tidal criterion: try to refactor since this is the heart of what I am adding to the method...
            f2body_i = mass[idx1] * pytreegrav.AccelTarget(np.atleast_2d(pos[idx1]), np.atleast_2d(pos[idx2]),
                                                        np.atleast_1d(mass[idx2]), h_target=np.atleast_1d(soft[idx1]),
                                                        h_source=np.atleast_1d(soft[idx2]), G=self.G)
            com_accel = (mass[idx1] * accel[idx1] + mass[idx2] * accel[idx2]) / (mass[idx1] + mass[idx2])
            f_tides = mass[idx1] * (accel[idx1] - com_accel) - f2body_i

            tidal_crit = (np.linalg.norm(f_tides) < np.linalg.norm(f2body_i)) or (not self.tides)
            if row[0] > 0 and (mult_total <= 4) and tidal_crit:
                print("adding {0}".format(mult_total))
                flag, ID_NEW = self._combine_binaries(row)
                if flag:
                    self._orbit_adjust_delete(ii, ID1, ID2)
                    self._orbit_adjust_add(ii, ID_NEW)
                return

    def _combine_binaries(self, row):
        sysIDs = self.get_system_ids_b
        sysID_max = np.max(sysIDs)

        filt = ~np.isin(sysIDs, row[-2:].astype(int).ravel())
        systems_new = self.systems[filt]
        ids = self.get_system_ids

        idx1 = np.where(sysIDs == row[-2])[0][0]
        idx2 = np.where(sysIDs == row[-1])[0][0]

        ##Can refactor: First conditional is not necessary, just don't filter out binaries in this case
        flag = 0
        if self.systems[idx1].multiplicity + self.systems[idx2].multiplicity > 4:
            systems_new = np.concatenate((systems_new, [self.systems[idx1], self.systems[idx2]]))
            flag = 0
        else:
            m1 = self.systems[idx1].mass
            m2 = self.systems[idx2].mass
            h1 = self.systems[idx1].soft
            h2 = self.systems[idx2].soft
            a_com = (m1 * self.systems[idx1].accel + m2 * self.systems[idx2].accel) / (m1 + m2)

            ss_new = system(row[4:7], row[7:10], row[10]+row[11], h1+h2, np.concatenate((ids[idx1], ids[idx2])), a_com, sysID_max+1)
            ss_new.add_orbit(self.systems[idx1].orbits)
            ss_new.add_orbit(self.systems[idx2].orbits)
            ss_new.add_orbit([row])
            systems_new = np.concatenate((systems_new, [ss_new]))
            flag = 1
        self.systems = systems_new
        return flag, sysID_max + 1
        # print(len(self.systems))


def main():
    snapshot_file = "snapshot_245.hdf5"
    # den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, tcgs, unit_base = load_data(snapshot_file, res_limit=1e-3)
    # cl = cluster(partpos, partvels, partmasses, partids)
    den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tcgs, unit_base = load_data(snapshot_file, res_limit=1e-3)
    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    partpos = partpos.astype(np.float64)
    partmasses = partmasses.astype(np.float64)
    partsink = partsink.astype(np.float64)

    accel_gas = pytreegrav.AccelTarget(partpos, xuniq, muniq, h_target=partsink, h_source=huniq, G=4.301e3)
    accel_stars = pytreegrav.Accel(partpos, partmasses, partsink, method='bruteforce', G=4.301e3)

    cl = cluster(partpos, partvels, partmasses, partsink, partids, accel_stars + accel_gas)
    with open("tmp_245_TidesTrue.p", "wb") as ff:
        pickle.dump(cl, ff)


if __name__ == "__main__":
    main()