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
sys.path.append("/home/aleksey/code/python")
from bash_command import bash_command as bc

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6

def snap_lookup(tmp_dat, pid, ID_COLUMN=0):
    tmp_idx = np.where(tmp_dat[:, ID_COLUMN].astype(int) == pid)[0][0]
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

    for ss in range(start_snap, end_snap + 1):
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

def get_unique_triples(r1, r2, start_snap, end_snap):
    """
    Get the unique binaries from a series of snapshots...

    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param int start_snap: Starting snapshot index
    :param int end_snap: Ending snapshot index

    :return: (i) List of unique binary ids (ii) List of the first time when they appear
    """
    uids = []
    first_appearance = []
    outer_idx = []
    inner_idx1 = []
    inner_idx2 = []
    for ss in range(start_snap, end_snap + 1):
        try:
            with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
                cl_a = pickle.load(ff)
        except FileNotFoundError:
            continue

        mults_a = np.array([sys1.multiplicity for sys1 in cl_a.systems])
        hiers_a = np.array([sys1.hierarchy for sys1 in cl_a.systems])
        ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)

        for jj, pp_set in enumerate(ids_a):
            if (mults_a[jj] == 3) and ~np.isin(pp_set, uids):
                uids.append(pp_set)
                first_appearance.append(ss)

                outer_idx.append(hiers_a[jj].pop())
                inner_idx1.append(hiers_a[jj][0][0])
                inner_idx2.append(hiers_a[jj][0][1])

    ##Also will need to pop outer index from stored hierarchy...
    return uids, np.array(first_appearance), outer_idx, inner_idx1, inner_idx2

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

def classify_triples(r1, r2, uids, first_snapshots):
    """
    Get the unique binaries from a series of snapshots...

    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param list uids: List of unique binary ids (from function get_unique_binaries)
    :param list first_snapshots: List of first snapshots where binaries appear (from function get_unique_binaries)

    :return: List of string encoding triple classification.
    :rtype: List
    """
    ##Split this into functions
    u_class = ['' for ii in range(len(uids))]
    for ii, pp_set in enumerate(uids):
        try:
            with open(r1 + "{0:03d}".format(first_snapshots[ii] - 1) + r2, "rb") as ff:
                cl = pickle.load(ff)
        except FileNotFoundError:
            continue

        pp = list(pp_set)
        ids_b = np.array([sys1.ids for sys1 in cl.systems], dtype=object)
        set_b = [set(row) for row in ids_b]
        ids_b_cat = np.concatenate(ids_b)

        ##System exists in the previous snapshot THIS SHOULD NOT HAPPEN -- MAYBE MARK WITH ERROR FLAG
        if np.in1d(pp_set, set_b)[0]:
            continue
        ##If no star index is present in previous snapshot we have a primordial triple formed
        if np.all(~np.in1d(pp, ids_b_cat)):
            u_class[ii] = 'p'
            continue
        ##If only some of the stars existed in the previous snapshot the origin of the triple is ambiguous
        if not np.all(np.in1d(pp, ids_b_cat)):
            u_class[ii] = 'sp'
            continue

        ##Check multiplicity of stars in the previous snapshot or subsequent analysis we only want to consider stars that existed in both snapshots--Not
        ##sure how to deal with this case perhaps we should add it a third "ambiguous" category
        tmp_mults = np.zeros(len(pp))
        ws = np.zeros(len(pp))
        for jj, ppp in enumerate(pp):
            w1 = np.where([np.in1d(ppp, row) for row in ids_b])
            ws[jj] = w1[0][0]
            tmp_mults[jj] = len(ids_b[w1[0][0]])
        ##If one of stars was a single consider a capture to have occurred, we consider a capture to have occured
        if np.isin([1], tmp_mults)[0]:
            u_class[ii] = 'c'
            continue

    return u_class

def create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap):
    """
    Get the a lookup table for parent system/orbit of each star

    :param r1 string: Bases of pickle file name
    :param r2 string: End of pickle file name
    :param start_snap int: Starting snapshot index
    :param end_snap int: Ending snapshot index

    :return: numpy array where columns are (i) snapshot (ii) star id (iii) index of parent system (iv) multiplicity
    mass (with and without gas), semimajor axis, eccentricity.
    :rtype: np.ndarray
    """
    lookup = []
    for ss in range(start_snap, end_snap + 1):
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
                    ecc1 = -1
                ##Using mass to identify particles -- In principle may not give a unique match...
                ##Perhaps use assert to check this assumption(!!!)
                else:
                    sel1 = np.isclose(m1, tmp_orb[:, 10:12])
                    sel1 = np.array([row[0] or row[1] for row in sel1])
                    sma1 = tmp_orb[sel1][0][0]
                    ecc1 = tmp_orb[sel1][0][1]
                lookup.append([ss, elem1, ii, len(ids_a[ii]), m1, w1_row[-1], sma1, ecc1])

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

##More flxible version of mult_filt
def mult_filt_id(ids, times, sys_lookup, mult_max=1):
    mults_filt = np.zeros(len(ids)).astype(bool)

    for ii in range(len(times)):
        sys_lookup_sel0 = sys_lookup[(sys_lookup[:, 1] == ids[ii]) & (sys_lookup[:, 0] < times[ii])]
        mults_filt[ii] = (np.all(sys_lookup_sel0[:, 3] <= mult_max))

    return mults_filt

def get_first_snap_idx(base_sink, start_snap, end_snap):
    """
    Get lookup table of the first snapshot index used
    """
    tmp_snap_idx = []
    for ss in range(start_snap, end_snap + 1):
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
        tmp_idx = tmp_sink[:, 0].astype(int)
        tmp_snap = [ss for ii in range(len(tmp_idx))]
        tmp_snap_idx.append(np.transpose([tmp_idx, tmp_sink[:, 1], tmp_sink[:, 2], tmp_sink[:, 3], tmp_snap]))

    tmp_snap_idx = np.vstack(tmp_snap_idx)
    tmp_uu, tmp_ui = np.unique(tmp_snap_idx[:, 0], return_index=True)
    first_snap_idx = tmp_snap_idx[tmp_ui]

    return first_snap_idx

def get_age_diff(bin_ids, first_snap_idx):
    """
    Age differences between the binary stars
    """
    delta_snap = np.zeros(len(bin_ids))
    for ii, pp in enumerate(bin_ids):
        row = list(pp)
        w1_row, w1 = snap_lookup(first_snap_idx, row[0])
        snap1 = first_snap_idx[w1, -1]

        w1_row, w1 = snap_lookup(first_snap_idx, row[1])
        snap2 = first_snap_idx[w1, -1]

        delta_snap[ii] = max(snap1, snap2) - min(snap1, snap2)
    return delta_snap

def get_age_diff_mult(r1, r2, end_snap, first_snap_idx):
    """
    Get difference between the final multiples
    """

    with open(r1 + "{0:03d}".format(end_snap) + r2, "rb") as ff:
        cl = pickle.load(ff)

    mults_b = np.array([sys1.multiplicity for sys1 in cl.systems])
    ids_b = np.array([sys1.ids for sys1 in cl.systems], dtype=object)

    end_ms = ids_b[mults_b > 2]
    delta_end_mults = np.zeros(len(end_ms))
    for ii, pp in enumerate(end_ms):
        row = list(pp)
        snaps = []
        for ppp in row:
            w1_row, w1 = snap_lookup(first_snap_idx, ppp)
            snap1 = first_snap_idx[w1, -1]
            snaps.append(snap1)
        delta_end_mults[ii] = np.max(snaps) - np.min(snaps)

    return delta_end_mults

def get_bin_histories(bin_ids, lookup):
    """
    Orbital elements of unique binaries *while they are togther* -- Possibly as part of a higher-order multiple...
    """
    bin_histories = []

    for row_s in bin_ids:
        id1, id2 = list(row_s)
        tmp_dat1 = lookup[lookup[:, LOOKUP_PID].astype(int) == id1]
        tmp_dat2 = lookup[lookup[:, LOOKUP_PID].astype(int) == id2]
        common_times = np.intersect1d(tmp_dat1[:, 0], tmp_dat2[:, 0], return_indices=True)
        tmp_dat1 = tmp_dat1[common_times[1]]
        tmp_dat2 = tmp_dat2[common_times[2]]
        tmp_filt = (tmp_dat1[:, LOOKUP_SMA] == tmp_dat2[:, LOOKUP_SMA])

        bin_histories.append(np.hstack((tmp_dat1[tmp_filt], tmp_dat2[tmp_filt])))

    bin_histories_stack = np.vstack(bin_histories)
    return bin_histories_stack

def get_fst(first_snapshot_idx, uids):
    fst_idx = np.zeros(len(uids)).astype(int)
    for ii, row in enumerate(uids):
        row_li = list(row)
        for tmp_item in row_li:
            tmp_snap1 = snap_lookup(first_snapshot_idx, tmp_item)[0][-1]
            fst_idx[ii] = max(tmp_snap1, fst_idx[ii])

    return fst_idx

def get_orbit_fst(base_sink, sys_lookup, fst_idx, uids, outer_idx):
    smas_all = np.ones(len(uids)) * np.inf
    for ii, row in enumerate(uids):
        tmp_snap = fst_idx[ii]
        if fst_idx[ii] < 100:
            tmp_snap = 100
            # continue
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(tmp_snap)))
        inner_part = np.array(list(row))
        inner_part = inner_part[~np.in1d(inner_part, [outer_idx[ii]])]
        inner_dat_all = []
        for pp in inner_part:
            ##NEED TO USE SYS_LOOKUP TABLE TO GET MASSES INCLUDING GAS...WONT
            m_gas = sys_lookup[(sys_lookup[:, LOOKUP_PID] == pp) & (sys_lookup[:, LOOKUP_SNAP] == tmp_snap)][0, LOOKUP_MTOT]
            inner_dat_all.append(snap_lookup(tmp_sink, pp)[0])
            inner_dat_all[-1][-1] = m_gas

        inner_dat_all = np.array(inner_dat_all)
        p1, v1, h1, m1_gas = get_com(np.atleast_2d(inner_dat_all))
        outer_dat_all = snap_lookup(tmp_sink, outer_idx[ii])[0]
        p2, v2, h2, m2_gas = outer_dat_all[1:4], outer_dat_all[4:7], outer_dat_all[7], outer_dat_all[-1]
        ##Once again including gas masses.
        m2_gas = sys_lookup[(sys_lookup[:, LOOKUP_PID] == outer_idx[ii]) & (sys_lookup[:, LOOKUP_SNAP] == tmp_snap)][0, LOOKUP_MTOT]

        tmp_orb = find_multiples_new2.get_orbit(p1, p2, v1, v2, m1_gas, m2_gas, h1=h1, h2=h2)
        ##Get just the semimajor axis for now...add more data later
        smas_all[ii] = tmp_orb[0]

    return smas_all

##Maybe should also add treatment of softening length
def get_com(dat):
    m1 = np.sum(dat[:, -1])
    pos1 = np.dot(dat[:, -1], dat[:, 1:4]) / m1
    vel1 = np.dot(dat[:, -1], dat[:, 4:7]) / m1
    h1 = np.sum(dat[:, 7])

    return pos1, vel1, h1, m1

def main():
    sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_2"
    base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/{0}/".format(sim_tag)
    r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/{0}/M2e4_snapshot_".format(sim_tag)
    r2 = sys.argv[1]
    base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
    r2_nosuff = r2.replace(".p", "")

    start_snap_sink = 48
    start_snap = 48
    end_snap = 423
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

    ##To add function which extracts FST INFO!!

    ##Binary classification -- Store output on disc. If it already exists then skip.
    try:
        classes = np.load(base + aa + "/classes.npz", allow_pickle=True)["arr_0"]
    except FileNotFoundError:
        classes = classify_binaries(r1, r2, bin_ids, first_snapshots)
        np.savez(base + aa + "/classes", classes)
    ##Table to lookup systems by particle ids
    try:
        sys_lookup = np.load(base + aa + "/system_lookup_table.npz", allow_pickle=True)["arr_0"]
    except FileNotFoundError:
        sys_lookup = create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap)
        np.savez(base + aa + "/system_lookup_table", sys_lookup)
    ##Tag marking the final fate of the binary.
    fate_tags = np.empty(len(bin_ids), dtype='S4')
    final_snap = np.empty(len(bin_ids), dtype=int)
    for ii, row in enumerate(bin_ids):
        fate_tags[ii] = get_fate(r1, r2, list(row), end_snap)
        tmp_idx = np.where(bin_ids_all == row)[0]
        final_snap[ii] = np.max(times_all[tmp_idx])
    np.savez(base + aa + "/fate_tags", fate_tags)
    np.savez(base + aa + "/bin_final_snap", final_snap)
    np.savez(base + aa + "/mults_filt", mult_filt(bin_ids, sys_lookup, ic))
    ##Refactor name(!) Too similar to first_snapshots?!
    first_snap_idx = get_first_snap_idx(base_sink, start_snap_sink, end_snap)
    delta_age = get_age_diff(bin_ids, first_snap_idx)
    np.savez(base + aa + "/delta_age_bins", delta_age)
    delta_age_mult = get_age_diff_mult(r1, r2, end_snap, first_snap_idx)
    np.savez(base + aa + "/delta_age_mults", delta_age_mult)
    bin_histories_stack = get_bin_histories(bin_ids, sys_lookup)
    np.savez(base + aa + "/bin_histories_stack", bin_histories_stack)
    ##Getting the final snapshot...
    with open(r1 + str(end_snap) + r2, "rb") as ff:
        cl_a = pickle.load(ff)

    mults_a = np.array([sys1.multiplicity for sys1 in cl_a.systems])
    ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)
    final_bins = ids_a[mults_a == 2]
    final_bin_histories_stack = get_bin_histories(final_bins, sys_lookup)
    np.savez(base + aa + "/final_bin_histories_stack", final_bin_histories_stack)

    final_bins_arr_id = np.array([np.where(bin_ids == row)[0][0] for row in final_bins])
    np.savez(base + aa + "/final_bins_arr_id", final_bins_arr_id)

    fst_idx = get_fst(first_snap_idx, bin_ids)
    np.savez(base + aa + "/fst", fst_idx)

    outer_idx = [list(row)[1] for row in bin_ids]
    first_snap_orb = get_orbit_fst(base_sink, sys_lookup, fst_idx, bin_ids, outer_idx)
    np.savez(base + aa + "/fst_sma", first_snap_orb)

    # uids, ic, outer_idx, inner_idx1, inner_idx2 = get_unique_triples(r1, r2, start_snap, end_snap)
    # fst_idx = get_fst(first_snap_idx, uids)
    # tri_class = classify_triples(r1, r2, uids, ic.astype(int))
    # np.savez(base + aa + "/tri_ic", ic)
    # np.savez(base + aa + "/tri_ids", uids)
    # np.savez(base + aa + "/tri_class", tri_class)
    # first_snap_orb = get_orbit_fst(base_sink, sys_lookup, fst_idx, uids, outer_idx)
    # np.savez(base + aa + "/tri_fst_sma", first_snap_orb)
    # np.savez(base + aa + "/tri_fst", fst_idx)
    # np.savez(base + aa + "/tri_mults_filt", mult_filt_id(outer_idx, ic, sys_lookup))
    # np.savez(base + aa + "/tri_mults_filt_b", mult_filt_id(inner_idx1, ic, sys_lookup, mult_max=2) & mult_filt_id(inner_idx2, ic, sys_lookup, mult_max=2))

    ##inner_idx1, inner_idx2...
#######################################################################################################################################################################



if __name__ == "__main__":
    main()