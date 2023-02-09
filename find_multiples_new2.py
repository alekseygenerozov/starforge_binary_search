import numpy as np
import h5py
from itertools import combinations
import pickle


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
    else:
        partpos = []
        partmasses = [0]
        partids = []
        partvels = [0, 0, 0]

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
    return den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, tcgs, unit_base


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
    # Ngrid1D = int(np.clip(np.min([dx, dy, dz]) / max_dist, 1, np.max([5, len(x) ** 0.33])));
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
    def __init__(self, p1, v1, m1, id1):
        self.pos = np.copy(p1)
        self.vel = np.copy(v1)
        self.mass = m1

        self.orbits = np.zeros((0, 14))
        self.ids = np.atleast_1d(id1)
        # self.child_pos
        # self.child_vel

    @property
    def multiplicity(self):
        return len(self.ids)

    def add_orbit(self, orb):
        self.orbits = np.concatenate((self.orbits, orb))


class cluster(object):
    def __init__(self, ps, vs, ms, ids):
        self.G = 4.301e3
        self.max_dist = 0.1
        self.systems = []
        for ii in range(len(ps)):
            self.systems.append(system(ps[ii], vs[ii], ms[ii], ids[ii]))
        self.systems = np.array(self.systems)
        self.orb_data = np.zeros((0, 14))
        self.regions = select_in_subregion(self.get_system_position, max_dist = self.max_dist)
        conv = False
        while not conv:
            systems_start = [ss.multiplicity for ss in self.systems]
            self.orb_data = np.zeros((0, 14))
            self._find_binaries_all()
            systems_end = [ss.multiplicity for ss in self.systems]
            conv = (systems_start == systems_end)
        # for ii in range(3):
        #     self.orb_data = np.zeros((0, 14))
        #     self.regions = select_in_subregion(self.get_system_position, max_dist = self.max_dist)
        #     self._find_binaries_all()
        #     self._combine_binaries()

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

    def _find_binaries_all(self):
        for ii in range(len(self.regions)):
            self.regions = select_in_subregion(self.get_system_position, max_dist=self.max_dist)
            self._find_bin_region(ii)

    def _find_bin_region(self, ii):
        region = self.regions[ii]
        pos = self.get_system_position[region]
        vel = self.get_system_vel[region]
        mass = self.get_system_mass[region]
        idx = np.array(range(len(self.systems)))[region]
        orb_all = []
        combos_all = np.array(list(combinations(list(range(len(pos))), 2)))
        for jj, combo in enumerate(combos_all):
            i = combo[0]
            j = combo[1]
            orb_all.append(np.concatenate((get_orbit(pos[i], pos[j], vel[i], vel[j], mass[i], mass[j], G=self.G), [idx[i], idx[j]])))
        if len(orb_all) < 1:
            return

        orb_all = np.array(orb_all)
        ens = -self.G*orb_all[:, 10]*orb_all[:, 11]/(2.*orb_all[:, 0])
        en_order = np.argsort(ens)
        # sma_order = np.argsort(orb_all[:, 0])
        orb_all = orb_all[en_order]
        combos_all = combos_all[en_order]

        bin_index = []
        for jj, combo in enumerate(combos_all):
            row = orb_all[jj]
            ##Need to the multiplicity check here?!
            # idx1 = combo[0]
            # idx2 = combo[1]
            idx1 = int(row[-2])
            idx2 = int(row[-1])
            mult_total = self.systems[idx1].multiplicity + self.systems[idx2].multiplicity
            if row[0] > 0 and ~np.isin(idx1, bin_index) and ~np.isin(idx2, bin_index) and (mult_total <= 4):
                print("adding {0}".format(mult_total))
                # self.orb_data.append(row)
                self.orb_data = np.vstack((self.orb_data, row))
                bin_index.append(idx1)
                bin_index.append(idx2)
                self._combine_binaries()
                return
                # self.orb_data = np.zeros((0, 14))
                # self.regions = select_in_subregion(self.get_system_position, max_dist=self.max_dist)
                # self._find_bin_region(ii)
        # self.orb_data = np.array(self.orb_data)


    def _combine_binaries(self):
        idx = np.array(range(len(self.systems)))
        filt = ~np.isin(idx, self.orb_data[:, -2:].astype(int).ravel())
        # print(len(self.systems[filt]))
        systems_new = self.systems[filt]
        # print(len(self.systems[filt]))

        masses = self.get_system_mass
        ids = self.get_system_ids
        for row in self.orb_data:
            idx1 = int(row[-2])
            idx2 = int(row[-1])
            if self.systems[idx1].multiplicity + self.systems[idx2].multiplicity > 4:
                # print("test1")
                systems_new = np.concatenate((systems_new, [self.systems[idx1], self.systems[idx2]]))
            else:
                # print("test2")
                ss_new = system(row[4:7], row[7:10], row[10]+row[11],  np.concatenate((ids[idx1], ids[idx2])))
                ss_new.add_orbit(self.systems[idx1].orbits)
                ss_new.add_orbit(self.systems[idx2].orbits)
                ss_new.add_orbit([row])
                systems_new = np.concatenate((systems_new, [ss_new]))
        self.systems = systems_new
        print(len(self.systems))


def main():
    snapshot_file = "snapshot_250.hdf5"
    den, x, m, h, u, b, v, t, fmol, fneu, partpos, partmasses, partvels, partids, tcgs, unit_base = load_data(snapshot_file, res_limit=1e-3)
    cl = cluster(partpos, partvels, partmasses, partids)
    with open("multiples.p", "wb") as ff:
        pickle.dump(cl, ff)


if __name__ == "__main__":
    main()