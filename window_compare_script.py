#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle
import copy

import cgs_const as cgs
sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
from analyze_multiples import snap_lookup
import find_multiples_new2, halo_masses_single_double_par
from find_multiples_new2 import cluster, system


def extract_inner_orb(hh, oo, id_coll, sma_coll, ecc_coll):
    h_copy = copy.copy(hh)
    ##Can we use orb_copy?? Would be safer
    orb_copy = copy.copy(oo)
    if type(h_copy)==list:
        p1 = (h_copy.pop())
        p2 = (h_copy.pop())
        orb1 = oo.pop()
        
    if (not type(p1)==list) and (not type(p2)==list):
        id_coll.append(str(np.sort([p1, p2])))
        sma_coll.append(orb1[0])
        ecc_coll.append(orb1[1])
    if type(p1)==list:
        extract_inner_orb(p1, oo, id_coll, sma_coll, ecc_coll)
    if type(p2)==list:
        extract_inner_orb(p2, oo, id_coll, sma_coll, ecc_coll)
        
def get_closest_pairs(sink_dat):
    """
    Get closest particle to each sink particle
    
    Get the separations and the relative velocities as well...
    """
    nparts = len(sink_dat)
    closest_id = []
    closest_dist = []
    closest_vrel = []
    for ii in range(nparts):
        part_i = sink_dat[ii]
        min_dist = np.inf
        min_id = -999
        min_vrel = -999
        for jj in range(nparts):
            if jj==ii:
                continue
            part_j = sink_dat[jj]
            sep = part_i[1:4] - sink_dat[jj, 1:4]
            tmp_dist = np.linalg.norm(sep)
            if tmp_dist < min_dist:
                min_dist = tmp_dist
                min_id = int(sink_dat[jj,0])
                ##Do we care about sign here??
                min_vrel = part_i[4:7] - sink_dat[jj, 4:7]
                min_vrel = np.dot(min_vrel, sep / np.linalg.norm(sep))
                
        closest_id.append(str(np.sort((int(min_id), int(part_i[0])))))
        closest_dist.append(min_dist)
        closest_vrel.append(min_vrel)
                
    return np.array(closest_id), np.array(closest_dist), np.array(closest_vrel)

def main():
    LOOKUP_SNAP = 0
    LOOKUP_PID = 1
    LOOKUP_ARR_IDX = 2
    LOOKUP_MULT = 3
    LOOKUP_MTOT = 4
    LOOKUP_SMA = 6
    LOOKUP_ECC = 7

    start_snap = 100
    end_snap = 489
    sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
    base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/".format(sim_tag)
    base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
    r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/M2e4_snapshot_".format(sim_tag)
    r2 = sys.argv[1]
    w_start = int(sys.argv[2])
    window = int(sys.argv[3])
    r_scale = cgs.pc
    v_scale = 100
    delta_t = 2.47e4 * cgs.year
    aa = "analyze_multiples_output_" + r2.replace(".p", "") + "/"
    lookup = np.load(base + aa + "system_lookup_table.npz")['arr_0']

    sinks_all = {}
    for ss in range(start_snap, end_snap + 1):
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
        sinks_all[ss] = (tmp_sink)

    ids_window = []
    dist_window = []
    vrel_window = []
    for ii in range(w_start, w_start + window + 1):
        tmp_sink = sinks_all[ii]
        closest_pairs_dat = get_closest_pairs(tmp_sink)
        ids_window = np.concatenate((ids_window, closest_pairs_dat[0]))
        dist_window = np.concatenate((dist_window, closest_pairs_dat[1]))
        vrel_window = np.concatenate((vrel_window, closest_pairs_dat[2]))

    ids_window_u, idx = np.unique(ids_window, return_index=True)
    counts_of_unique = np.ones(len(ids_window_u)) * -1
    for ii, uu in enumerate(ids_window_u):
        cc = np.count_nonzero(ids_window == uu)
        counts_of_unique[ii] = cc

    bin_ck = np.zeros(len(ids_window_u)).astype(bool)
    bin_ck_lax = np.zeros(len(ids_window_u)).astype(bool)

    for ii, uu in enumerate(ids_window_u):
        if counts_of_unique[ii] < 2 * (window + 1):
            continue
        bin_ck_lax[ii] = True
        filt1 = ids_window == uu
        vs = vrel_window[filt1]
        ds = dist_window[filt1]

        bin_ck[ii] = (np.max(ds) - np.min(ds)) * r_scale < 0.5 * np.mean(np.abs(vs)) * v_scale * delta_t * window

    ss = (2 * w_start + window) // 2
    lookup_sel = lookup[lookup[:, LOOKUP_SNAP] == ss]

    with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
        cl_a = pickle.load(ff)
    ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)
    orbs_a = np.array([sys1.orbits for sys1 in cl_a.systems], dtype=object)
    hiers_a = np.array([sys1.hierarchy for sys1 in cl_a.systems], dtype=object)

    id_coll = []
    sma_coll = []
    ecc_coll = []
    for ii in range(len(ids_a)):
        if len(ids_a[ii]) == 1:
            continue
        extract_inner_orb(hiers_a[ii], list(np.atleast_2d(orbs_a[ii])), id_coll, sma_coll, ecc_coll)

    id_coll = np.array(id_coll)
    sma_coll = np.array(sma_coll)
    ecc_coll = np.array(ecc_coll)
    np.savetxt("id_coll{0}_{1}".format(r2.replace(".p", ""), w_start), id_coll, fmt="%s")

    in_new_bins = np.in1d(id_coll, ids_window_u[bin_ck])
    in_old_bins = np.in1d(ids_window_u[bin_ck], id_coll)

    fig, ax = plt.subplots()
    ax.set_xlim(1, 1e5)
    ax.set_xlabel('a [au]')
    ax.set_ylabel('e')
    ax.set_xscale('log')

    ax.scatter(sma_coll * cgs.pc / cgs.au, ecc_coll)
    ax.scatter(sma_coll[in_new_bins] * cgs.pc / cgs.au, ecc_coll[in_new_bins], c='r',\
               label='ID with New Method', alpha=0.5, marker="s")
    ax.legend()
    fig.savefig("comp{0}_{1}_{2}.pdf".format(r2.replace(".p", ""), w_start, window))

    ##CHECK FOR NEW BINARIES IDENTIFIED WITH THE NEW METHED THAT WERE NOT IDENTIFIED WITH THE OLD ONE...
    # print(ids_window_u[bin_ck][~in_old_bins])
    frac1 = len(ids_window_u[bin_ck][in_old_bins]) / len(ids_window_u[bin_ck])
    frac2 = len(id_coll[in_new_bins]) / len(id_coll)
    filt_close = sma_coll * cgs.pc / cgs.au < 1000
    frac3 = len(id_coll[filt_close][in_new_bins[filt_close]]) / len(id_coll[filt_close])

    # with open("frac_{0}_{1}".format(w_start), "a") as ff:
    print("{0} {1} {2}\n".format(frac1, frac2, frac3))

    with open("window_info{0}_{1}".format(r2.replace(".p", ""), w_start), "w") as ff:
        for tmp_id in ids_window_u[bin_ck][~in_old_bins]:
            ff.write(tmp_id + " ")
            tmp_id_num = tmp_id.replace("[", "").replace("]", "").split()
            for idx in tmp_id_num:
                row = lookup_sel[lookup_sel[:, LOOKUP_PID] == int(idx)]
                ai, mm = row[0][LOOKUP_ARR_IDX], row[0][LOOKUP_MULT]
                ff.write(str(ai) + " ")
                ff.write(str(mm) + " ")
            ff.write("\n")

    np.savetxt("non_detect{0}_{1}".format(r2.replace(".p", ""), w_start),
               id_coll[filt_close][~in_new_bins[filt_close]], fmt="%s")

    # print("Fraction of window bins identified in old method:", )
    # print("Fraction of old method binaries identified in window search:", )
    # print("Fraction of <1000 au old method binaries identified in window search:",)
    #

if __name__ == "__main__":
    main()





