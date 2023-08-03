import glob
import numpy as np
import matplotlib.pyplot as plt
##See if we can do the filtering in a cleaner/simpler way np.in1d can be a little uninituitive in some cases
import sys
import pickle

sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import find_multiples_new2, halo_masses_single_double_par
from find_multiples_new2 import cluster, system
import h5py
from bash_command import bash_command as bc


def snap_lookup(tmp_dat, pid):
    tmp_idx = np.where(tmp_dat[:, 0].astype(int) == pid)[0][0]
    return tmp_dat[tmp_idx], tmp_idx


def get_energy_from_sink_table(tmp_dat, row_idx):
    tmp_dat_select = tmp_dat[row_idx]
    tmp_pos = tmp_dat_select[:, 1:4]
    tmp_vel = tmp_dat_select[:, 4:7]
    tmp_h = tmp_dat_select[:, -2]
    tmp_mass = tmp_dat_select[:, -1]

    pe1 = halo_masses_single_double_par.PE(tmp_pos, tmp_mass, tmp_h)
    ke1 = halo_masses_single_double_par.KE(tmp_pos, tmp_mass, tmp_vel, np.zeros(len(tmp_pos)))

    return pe1 + ke1


def get_fate(r1, r2, row, ss):
    """
    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param list row: List (or other subscriptable) with binary elements
    :param int ss: Snapshot index

    :return: Fate of binary (specified by row) in snapshot number ss. (i) d = At least on star deleted
    (ii) s[2-4] = In system of multiplicity i (s2 means system is surviving as a binary) (iii) i = ionized
    (iv) mm = Stars are in separate multiples (v) ms = One star is in a single, while the other is in a multiple
    :rtype: string
    """
    with open(r1 + "{0:03d}".format(int(ss)) + r2, "rb") as ff:
        cl = pickle.load(ff)
    mults_a = np.array([sys1.multiplicity for sys1 in cl.systems])
    ids_a = np.array([sys1.ids for sys1 in cl.systems], dtype=object)

    try:
        idx1 = np.where(np.concatenate([np.isin([row[0]], tmp_row) for tmp_row in ids_a]))[0][0]
        idx2 = np.where(np.concatenate([np.isin([row[1]], tmp_row) for tmp_row in ids_a]))[0][0]
    except IndexError:
        return 'd'

    if idx1 == idx2:
        mult_sys = len(ids_a[idx1])
        return ('s' + str(mult_sys))
    else:
        if len(ids_a[idx1]) == 1 and len(ids_a[idx2]) == 1:
            return 'i'
        elif len(ids_a[idx1]) > 1 and len(ids_a[idx2]) > 1:
            return 'mm'
        elif (len(ids_a[idx1]) == 1 and len(ids_a[idx2]) == 2) or (len(ids_a[idx1]) == 2 and len(ids_a[idx2]) == 1):
            return 'bs'
        elif (len(ids_a[idx1]) == 1 and len(ids_a[idx2]) > 2) or (len(ids_a[idx1]) > 2 and len(ids_a[idx2]) == 1):
            return 'ms'

# def get_binary_lifetime(r1, r2, row, first_appearance, end_snap):
#     """
#     Get the unique binaries from a series of snapshots...
#
#     :param string r1: Bases of pickle file name
#     :param string r2: End of pickle file name
#     :param set row: Set with the system ids
#     :param int first_appearance: First snapshot where binary appears
#     :param int end_snap: Ending snapshot of the simulation
#
#     :return: Final snaphot in which system appears
#     :rtype: int
#     """


def get_unique_binaries(r1, r2, start_snap, end_snap):
    """
    Get the unique binaries from a series of snapshots...

    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param int start_snap: Starting snapshot index
    :param int end_snap: Ending snapshot index

    :return: (i) List of unique binary ids (ii) Array with the following
     columns (a) Snapshot of first appearance
     (b) Iniitial smas (c) Initial eccentricities (d) Initial com position
     (e) Initial masses (f) Initial separation
    :rtype: Tuple
    """
    bin_ids = []
    first_appearance = []
    sma_i = []
    ecc_i = []
    mass_i1 = []
    mass_i2 = []
    pos_ix = []
    pos_iy = []
    pos_iz = []
    sep_i = []

    bin_ids_all = []
    times_all = []
    nsys = []

    for ss in range(start_snap, end_snap):
        try:
            with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
                cl_a = pickle.load(ff)
        except FileNotFoundError:
            continue

        mults_a = np.array([sys1.multiplicity for sys1 in cl_a.systems])
        ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)
        bin_ids_all.append(ids_a[mults_a == 2])
        times_all.append(np.ones(ids_a[mults_a == 2].size) * ss)
        nsys.append([ss, len(ids_a), len(ids_a[mults_a == 2]), len(ids_a[mults_a == 3]), len(ids_a[mults_a == 4])])

        for jj, pp_set in enumerate(ids_a):
            if (mults_a[jj] == 2) and ~np.isin(pp_set, bin_ids):
                bin_ids.append(pp_set)
                first_appearance.append(ss)
                sma_i.append(cl_a.systems[jj].orbits[0, 0])
                ecc_i.append(cl_a.systems[jj].orbits[0, 1])
                pos_ix.append(cl_a.systems[jj].orbits[0, 4])
                pos_iy.append(cl_a.systems[jj].orbits[0, 5])
                pos_iz.append(cl_a.systems[jj].orbits[0, 6])
                mass_i1.append(cl_a.systems[jj].orbits[0, 10])
                mass_i2.append(cl_a.systems[jj].orbits[0, 11])
                sep_i.append(np.linalg.norm(np.diff(cl_a.systems[jj].sub_pos, axis=0)))

    return bin_ids, np.transpose((first_appearance, sma_i, ecc_i, pos_ix, pos_iy, pos_iz,
                                  mass_i1, mass_i2, sep_i)), np.concatenate(bin_ids_all), np.concatenate(times_all),\
        nsys

def classify_binaries(r1, r2, bin_ids, first_snapshots):
    """
    Get the unique binaries from a series of snapshots...

    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param list bin_ids: List of unique binary ids (from function get_unique_binaries)
    :param list first_snapshots: List of first snapshots where binaries appear (from function get_unique_binaries)

    :return: List of string encoding binary classification.
    :rtype: List
    """
    ##Split this into functions
    bin_class = ['' for ii in range(len(bin_ids))]
    for ii, pp_set in enumerate(bin_ids):
        try:
            with open(r1 + "{0:03d}".format(first_snapshots[ii] - 1) + r2, "rb") as ff:
                cl = pickle.load(ff)
        except FileNotFoundError:
            continue

        pp = list(pp_set)
        mults_b = np.array([sys1.multiplicity for sys1 in cl.systems])
        ids_b = np.array([sys1.ids for sys1 in cl.systems], dtype=object)
        set_b = [set(row) for row in ids_b]
        ids_b_cat = np.concatenate(ids_b)

        ##System exists in the previous snapshot THIS SHOULD NOT HAPPEN
        if np.in1d(pp_set, set_b)[0]:
            continue
        ##If neither star index is present in previous snapshot we have a primordial binary formed
        if np.all(~np.in1d(pp, ids_b_cat)):
            bin_class[ii] = 'p'
            continue
        ##If only some of the stars existed in the previous snapshot the origin of the binary is ambiguous
        if not np.all(np.in1d(pp, ids_b_cat)):
            bin_class[ii] = 'sp'
            continue

        ##Check multiplicity of stars in the previous snapshotor subsequent analysis we only want to consider stars that existed in both snapshots--Not
        ##sure how to deal with this case perhaps we should add it a third "ambiguous" category
        tmp_mults = np.zeros(len(pp))
        ws = np.zeros(len(pp))
        for jj, ppp in enumerate(pp):
            w1 = np.where([np.in1d(ppp, row) for row in ids_b])
            ws[jj] = w1[0][0]
            tmp_mults[jj] = len(ids_b[w1[0][0]])
        ##If all of the multiple stars were singles, we consider a capture to have occured
        if np.all(tmp_mults == 1):
            bin_class[ii] = 'c'
            continue
        w_unique = np.unique(ws)
        if len(w_unique) == 1:
            bin_class[ii] = 'diss'
            continue
        bin_class[ii] = 'ex'

    return bin_class

def create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap):
    """
    Get the unique binaries from a series of snapshots...

    :param r1 string: Bases of pickle file name
    :param r2 string: End of pickle file name
    :param start_snap int: Starting snapshot index
    :param end_snap int: Ending snapshot index

    :return: numpy array where columns are (i) snapshot (ii) star id (iii) index of parent system (iv) multiplicity
    :rtype: np.ndarray
    """
    lookup = []
    for ss in range(start_snap, end_snap):
        try:
            with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
                cl = pickle.load(ff)
        except FileNotFoundError:
            continue
        ids_a = np.array([sys1.ids for sys1 in cl.systems], dtype=object)
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))

        for ii in range(len(ids_a)):
            for jj, elem1 in enumerate(ids_a[ii]):
                m1 = cl.systems[ii].sub_mass[jj]
                tmp_orb = cl.systems[ii].orbits
                w1_row, w1_idx = snap_lookup(tmp_sink, elem1)
                if len(tmp_orb) == 0:
                    sma1 = -1
                ##Using mass to identify particles -- In principle may not give a unique match...
                else:
                    sel1 = np.isclose(m1, tmp_orb[:, 10:12])
                    sel1 = np.array([row[0] or row[1] for row in sel1])
                    sma1 = tmp_orb[sel1][0][0]
                lookup.append([ss, elem1, ii, len(ids_a[ii]), m1, w1_row[-1], sma1])

    return np.array(lookup)

def mult_filt(bin_ids, sys_lookup, ic):
    """
    Checking to see if multiples had previously been single prior to their first appearance

    :param bin_ids Array-like: List of binary ids to check
    :param sys_lookup Array-like: Lookup table for stellar properties
    :param ic Array-like: Lookup table for initial binary properties

    :return: Array of booleans. True=Multiple stars had previously been in multiple
    :rtype: np.ndarray
    """
    t_first = ic[:, 0]
    mults_filt = np.zeros(len(bin_ids)).astype(bool)
    for ii, row in enumerate(bin_ids):
        row_list = list(row)
        sys_lookup_sel0 = sys_lookup[(sys_lookup[:, 1] == row_list[0]) & (sys_lookup[:, 0] < t_first[ii])]
        sys_lookup_sel1 = sys_lookup[(sys_lookup[:, 1] == row_list[1]) & (sys_lookup[:, 0] < t_first[ii])]
        mults_filt[ii] = (np.all(sys_lookup_sel0[:, 3] == 1) and np.all(sys_lookup_sel1[:, 3] == 1))

    return mults_filt

def main():
    sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
    base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/".format(sim_tag)
    r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/M2e4_snapshot_".format(sim_tag)
    r2 = sys.argv[1]
    base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
    r2_nosuff = r2.replace(".p", "")

    start_snap = 100
    end_snap = 200
    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    bc.bash_command("mkdir " + base + aa)
    with open(base + aa + "/mult_data_path", "w") as ff:
        ff.write(r1 + "\n")
        ff.write(r2 + "\n")
    ##Unique binaries
#######################################################################################################################################################################
    ##Binary ids and initial conditions
    try:
        bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
        ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
        bin_ids_all = np.load(base + aa + "/bin_ids_all.npz", allow_pickle=True)['arr_0']
        times_all = np.load(base + aa + "/times_all.npz", allow_pickle=True)['arr_0']
        nsys = np.load(base + aa + "/nsys.npz", allow_pickle=True)['arr_0']
    except FileNotFoundError:
        bin_ids, ic, bin_ids_all, times_all, nsys = get_unique_binaries(r1, r2, start_snap, end_snap)
        np.savez(base + aa + "/unique_bin_ids", bin_ids)
        np.savez(base + aa + "/ic", ic)
        np.savez(base + aa + "/bin_ids_all", bin_ids_all)
        np.savez(base + aa + "/times_all", times_all)
        np.savez(base + aa + "/nsys", nsys)
    first_snapshots = ic[:, 0].astype(int)

    ##Binary classification -- Store output on disc. If it already exists then skip.
    try:
        classes = np.load(base + aa + "/classes.npz", allow_pickle=True)
    except FileNotFoundError:
        classes = classify_binaries(r1, r2, bin_ids, first_snapshots)
        np.savez(base + aa + "/classes", classes)
    ##Table to lookup systems by particle ids
    try:
        sys_lookup = np.load(base + aa + "/system_lookup_table.npz", allow_pickle=True)
    except FileNotFoundError:
        sys_lookup = create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap)
        np.savez(base + aa + "/system_lookup_table", sys_lookup)
    ##Tag marking the final fate of the binary.
    fate_tags = np.empty(len(bin_ids), dtype='S4')
    final_snap = np.empty(len(bin_ids), dtype=int)
    for ii, row in enumerate(bin_ids):
        fate_tags[ii] = get_fate(r1, r2, list(row), end_snap - 1)
        tmp_idx = np.where(bin_ids_all == row)[0]
        final_snap[ii] = np.max(times_all[tmp_idx])
    np.savez(base + aa + "/fate_tags", fate_tags)
    np.savez(base + aa + "/bin_final_snap", final_snap)
    np.savez(base + aa + "/mults_filt", mult_filt(bin_ids, sys_lookup, ic))
#######################################################################################################################################################################



if __name__ == "__main__":
    main()