import numpy as np
import pytreegrav
from matplotlib import pyplot as plt
import sys
import pickle

sys.path.append("/home/aleksey/code/python/")
sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
from analyze_multiples import snap_lookup
import find_multiples_new2
import cgs_const as cgs
import starforge_constants as sfc

import tracemalloc
tracemalloc.start()
import gc
import glob

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6
LOOKUP_ECC = 7

# def calculate_acceleration(path_lookup, tag1, tag2):
nclose = 10

def subtract_path(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff

def path_divide(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones(len(p1)) * np.inf
    filt = (~np.isinf(p1)) & (~np.isinf(p2))
    diff[filt] = p1[filt] / p2[filt]

    return diff

def max_w_infinite(p1):
    if np.all(np.isinf(p1)):
        return np.inf
    else:
        return np.max(p1[~np.isinf(p1)])

##READING IN AUXILIARY DATA TABLES...############################################################################################
sim_tag = f"M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{sys.argv[1]}"
base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{sys.argv[1]}/"
r2 = sys.argv[2].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
lookup = np.load(base + aa + "/system_lookup_table.npz")['arr_0']
bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
# final_bins_arr_id = np.load(base + aa + "/final_bins_arr_id.npz")["arr_0"]
bin_class = np.load(base + aa + "/classes.npz", allow_pickle=True)['arr_0']
##Tides data
# tides_norm_series = np.load(base + aa + "//tides_norm_series.npz")["arr_0"]
#################################################################################################################################
##Getting starting and ending snapshots
snaps = [xx.replace(base_sink, "").replace(".sink", "") for xx in glob.glob(base_sink + "*.sink")]
snaps = np.array(snaps).astype(int)

##Get snapshot numbers automatically
start_snap_sink = min(snaps)
start_snap = min(snaps)
end_snap = max(snaps)
#################################################################################################################################

t_first = ic[:, 0]
conv = cgs.pc / cgs.au
snap_interval = 2.47e4
##TRANSFORMATION TO PARTICLE ############################################################################################
sinks_all = []
ts = []
tags = []
accels = []
nsinks = np.zeros(end_snap + 1)
for ss in range(start_snap, end_snap + 1):
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
    sinks_all.append(tmp_sink)
    # tmp_accel_stars = pytreegrav.Accel($$$, $$$, $$$, theta=0.5, G=sfc.GN, method='bruteforce')
    # accels.append(tmp_accel_stars)

    ts.append(ss * np.ones(len(tmp_sink)))

sinks_all = np.vstack(sinks_all)
ts = np.concatenate(ts)
ts.shape = (-1, 1)
sinks_all = np.hstack((ts, sinks_all))
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))

# sinks_all = np.hstack((ts, sinks_all, accels))
# sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m", "ax", "ay", "az"))

tags = ["{0}_{1}".format(sinks_all[ii, 1], sinks_all[ii, 0]) for ii in range(len(ts))]
sinks_all = sinks_all[np.argsort(tags)]
tags2 = ["{0}_{1}".format(lookup[ii, 1], lookup[ii, 0]) for ii in range(len(ts))]
lookup_sorted = lookup[np.argsort(tags2)]
sinks_all = np.hstack((sinks_all, lookup_sorted[:,[2, LOOKUP_MTOT, LOOKUP_SMA, LOOKUP_ECC]]))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
assert(np.all(np.array(tags)[np.argsort(tags)] == np.array(tags2)[np.argsort(tags2)]))
######Saving a path for each particle
utags = np.unique(sinks_all[:, 1])
utags_str = utags.astype(int).astype(str)
utimes = np.unique(sinks_all[:, 0])
path_lookup = {}
path_lookup_times = {}
for ii, uu in enumerate(utags):
    tmp_sel = sinks_all[sinks_all[:, 1] == uu]
    tmp_path1 = np.ones((end_snap + 1, len(sink_cols))) * np.inf
    tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel

    path_lookup[utags_str[ii]] = tmp_path1

for ii, uu in enumerate(utimes):
    tmp_sel = sinks_all[sinks_all[:, 0] == uu]
    path_lookup_times[int(uu)] = tmp_sel

with open(base + aa + "/path_lookup.p", "wb") as ff:
    pickle.dump(path_lookup, ff)

with open(base + aa + "/path_lookup_times.p", "wb") as ff:
    pickle.dump(path_lookup_times, ff)

mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]
##MAKING PLOTS OF TRAJECTORIES FOR ALL OF THE BINARIES ...############################################################################################
fig_idx = 0
dens_series = np.zeros((len(bin_ids), end_snap + 1))
cross_sections = np.zeros((len(bin_ids), end_snap + 1))
force_series = np.zeros((len(bin_ids), end_snap + 1))
rms_mass = np.zeros((len(bin_ids), end_snap + 1))
mean_mass = np.zeros((len(bin_ids), end_snap + 1))
sigma_series = np.zeros((len(bin_ids), end_snap + 1))
halo_snap = np.zeros(len(bin_ids))
closest = []

print(base + aa + "closest.p")
snapshot1 = tracemalloc.take_snapshot()

for jj in range(len(bin_ids[:5])):
    print(jj)
    # if not np.isin(jj, final_bins_arr_id):
    #     continue

    tmp_row = np.array(list(bin_ids[jj])).astype(str)
    tmp_class = bin_class[jj]
    ##Getting com over time for pair -- make sure that the replacement here will not cause errors
    p1_raw = path_lookup[tmp_row[0]]
    ms1 = p1_raw[:, mcol]
    ms1[np.isinf(ms1)] = 1
    ms1.shape = (-1, 1)
    p2_raw = path_lookup[tmp_row[1]]
    ms2 = p2_raw[:, mcol]
    ms2[np.isinf(ms2)] = 1
    ms2.shape = (-1, 1)
    coms = (p1_raw[:, pxcol:pzcol + 1] * ms1 + p2_raw[:, pxcol:pzcol + 1] * ms2) / (ms1 + ms2)
    ##Positions in the evolving com frame
    p1 = subtract_path(p1_raw[:, pxcol:pzcol + 1], coms)  # [100:]
    p2 = subtract_path(p2_raw[:, pxcol:pzcol + 1], coms)  # [100:]
    psep = subtract_path(p1_raw[:, pxcol:pzcol + 1], p2_raw[:, pxcol:pzcol + 1])
    psep = np.sum((psep * psep), axis=1)**.5
    tmp_filt = ~np.isinf(p1[:, 0])
    ##Filter to exclude data where binary separation falls below the softening length
    pfilt = np.where(~np.isinf(psep) & (psep > 2e-4))[0]
    t_avg = np.inf
    t_max = np.inf
    ##Mean tidal force for binaries while separation is greater than the softening length...
    # t_series = tides_norm_series[jj]
    # if len(pfilt) > 0:
    #     pfilt = pfilt[-1]
    #     tmp_t_series = t_series[:pfilt + 1]
    #     t_avg = np.mean(tmp_t_series[~np.isinf(tmp_t_series)])
    #     t_max = np.max(tmp_t_series[~np.isinf(tmp_t_series)])


    sys1_info = lookup[int(tmp_row[0]) == lookup[:, LOOKUP_PID].astype(int)]
    sys2_info = lookup[int(tmp_row[1]) == lookup[:, LOOKUP_PID].astype(int)]

    ##Getting initial semimajor axis -- need to do this ringamarole in case the
    ##stars start unbound from each other...
    p1_raw_fst = p1_raw[tmp_filt][0]
    p2_raw_fst = p2_raw[tmp_filt][0]
    tmp_orb = find_multiples_new2.get_orbit(p1_raw_fst[pxcol:pzcol+1], p2_raw_fst[pxcol:pzcol+1],\
                                            p1_raw_fst[vxcol:vzcol+1], p2_raw_fst[vxcol:vzcol+1],\
                                            p1_raw_fst[mtotcol], p2_raw_fst[mtotcol],\
                                           p1_raw_fst[hcol], p2_raw_fst[hcol])


    # coms = np.ones((end_snap + 1, 3)) * np.inf
    ##Get orbits(!!!)
    ##Just have to do this once at the FST(!)

    ##Have to take care of nans here...
    closest_approaches = np.zeros(len(utags))
    closest_approaches_norm = np.zeros(len(utags))
    closest_approaches_time = np.zeros(len(utags))
    closest_approaches_norm_time = np.zeros(len(utags))

    path_diff_all = []
    mass_all = []
    vel_all_x = []
    vel_all_y = []
    vel_all_z = []
    u_tags_exclude = []
    for ii, uu in enumerate(utags_str):
        #Want only closest approach of stars external to the binary.
        if uu in tmp_row:
            continue
        u_tags_exclude.append([uu] * len(path_lookup[uu]))
        ##Displacement from binary com
        path_diff = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], coms)
        path_diff = np.sum(path_diff * path_diff, axis=1)**.5
        path_diff_all.append(path_diff)
        mass_all.append(path_lookup[uu][:, mtotcol])
        vel_all_x.append(path_lookup[uu][:, vxcol])
        vel_all_y.append(path_lookup[uu][:, vycol])
        vel_all_z.append(path_lookup[uu][:, vzcol])
        ##Closest approach for this particle
        order = np.argsort(path_diff)
        closest_approaches_time[ii] = path_lookup[uu][order[0]][0] * snap_interval
        closest_approaches[ii] = path_diff[order[0]] * conv
        ##Get separation normalized by binary separation
        path_diff_norm = path_divide(path_diff, psep)
        order_norm = np.argsort(path_diff_norm)
        ##Time where (normalized) closest approach occurs
        ##Time where (normalized) closest approach occurs
        closest_approaches_norm_time[ii] = path_lookup[uu][order_norm[0]][0] * snap_interval
        ##Normalized closest approach
        closest_approaches_norm[ii] = path_diff_norm[order_norm[0]]
        closest_approaches_norm[ii] = path_diff[order[0]]
        ##FORCE ON EACH PARTICLE
        ##NEED TO CALL PYTREEGRAV

    ##Transpose distance and mass array so that the rows are times.
    path_diff_all = np.array(path_diff_all).T
    mass_all = np.array(mass_all).T
    vel_all_x = np.array(vel_all_x).T
    vel_all_y = np.array(vel_all_y).T
    vel_all_z = np.array(vel_all_z).T
    u_tags_exclude = np.array(u_tags_exclude).T
    ##Time series of maximum force (NB WITHOUT SOFTENING)
    tmp_force = path_divide(mass_all.ravel(), (path_diff_all**2.).ravel())
    tmp_force.shape = mass_all.shape
    force_series[jj] = [max_w_infinite(row) for row in tmp_force]
    ##Ordering particles by distance at every snapshot
    path_diff_all_order = np.argsort(path_diff_all, axis=1)
    path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
    mass_all = np.take_along_axis(mass_all, path_diff_all_order, axis=1)
    vel_all_x = np.take_along_axis(vel_all_x, path_diff_all_order, axis=1)
    vel_all_y = np.take_along_axis(vel_all_y, path_diff_all_order, axis=1)
    vel_all_z = np.take_along_axis(vel_all_z, path_diff_all_order, axis=1)
    u_tags_exclude = np.take_along_axis(u_tags_exclude, path_diff_all_order, axis=1)
    ##All rows in u_tags_exclude are the same (UNNECESSARY -- TO FIX). For now just take the first row...
    closest.append(u_tags_exclude[:, 0])
    ##Compute time series of the local density for this binary, using the 32 closes particles...
    dens_series[jj] =  path_divide(np.ones(len(path_diff_all)) * nclose, path_diff_all[:, nclose-1] ** 3.)
    mean_mass[jj] = np.mean(mass_all[:, :nclose], axis=1)
    rms_mass[jj] = np.mean(mass_all[:, :nclose]**2., axis=1)**.5
    ##Need to calculate the velocity dispersions...
    v_close_x = vel_all_x[:, :nclose]
    sigma_x2 = np.mean(v_close_x**2., axis=1) - np.mean(v_close_x, axis=1)**2.
    v_close_y = vel_all_y[:, :nclose]
    sigma_y2 = np.mean(v_close_y**2., axis=1) - np.mean(v_close_y, axis=1)**2.
    v_close_z = vel_all_z[:, :nclose]
    sigma_z2 = np.mean(v_close_z**2., axis=1) - np.mean(v_close_z, axis=1)**2.
    sigma_2 = sigma_x2 + sigma_y2 + sigma_z2
    sigma_2[np.isnan(sigma_2)] = np.inf
    cross_sections[jj] = path_divide(psep * sfc.GN * (p1_raw[:, mcol] + p2_raw[:, mcol]), (sigma_2))
    sigma_series[jj] = sigma_2**.5
    ##Minimum closest approach over all particles
    order = np.argsort(closest_approaches)
    closest_tags = utags_str[order]
    closest_approaches = closest_approaches[order]
    closest_approaches_time = closest_approaches_time[order]
    #Minimum normalized closest approach over all particles
    order_norm = np.argsort(closest_approaches_norm)
    closest_tags_norm = utags_str[order_norm]
    closest_approaches_norm = closest_approaches_norm[order_norm]
    closest_approaches_norm_time = closest_approaches_norm_time[order_norm]

    ##Halo masses
    tmp_halo_mass = sys1_info[:, LOOKUP_MTOT] - sys1_info[:, LOOKUP_M]
    halo_snap1_list = np.where(tmp_halo_mass > 0.01 * sys1_info[:, LOOKUP_M])[0]
    tmp_halo_mass = sys2_info[:, LOOKUP_MTOT] - sys2_info[:, LOOKUP_M]
    halo_snap2_list = np.where(tmp_halo_mass > 0.01 * sys2_info[:, LOOKUP_M])[0]

    ##Last snapshot with significant gas(!)
    halo_snap1 = 0
    halo_snap2 = 0
    if len(halo_snap1_list) > 0:
        halo_snap1 = sys1_info[halo_snap1_list[-1]][LOOKUP_SNAP]
    if len(halo_snap2_list) > 0:
        halo_snap2 = sys2_info[halo_snap2_list[-1]][LOOKUP_SNAP]
    halo_snap[jj] = max(halo_snap1, halo_snap2)

    pfilt = np.where(~np.isinf(psep) & (psep > 2e-4))[0]
    # t_avg = np.inf
    # t_max = np.inf
    # snap_max = np.inf

    # snapshot2 = tracemalloc.take_snapshot()
    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    #
    # print(f"Iteration {jj}")
    # for stat in top_stats[:10]:
    #     print(stat)
    # gc.collect()

    # snapshot1 = snapshot2  # Update snapshot for next iteration

    #Mean tidal force for binaries while separation is greater than the softening length...
    # t_series = tides_norm_series[jj]
    # if len(pfilt) > 0:
    #     pfilt = pfilt[-1]
    #     if halo_snap[jj] + 1 <= pfilt:
    #         tmp_t_series = t_series[int(halo_snap[jj] + 1):pfilt + 1]
    #         t_avg = np.mean(tmp_t_series[~np.isinf(tmp_t_series)])
    #         t_max = np.max(tmp_t_series[~np.isinf(tmp_t_series)])
    #         snap_max = np.where(t_series == t_max)[0][0]

    # if np.isinf(t_max) or t_max >= 0.01:
    #     continue
    fig, axs = plt.subplots(figsize=(20, 20), nrows=2, ncols=2)
    # fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    delta = np.abs(p1[tmp_filt][0] - p2[tmp_filt][0]) * conv

    ax = axs[0,0]
    ax.set_xlabel("x [au]")
    ax.set_ylabel("y [au]")
    ax.set_xlim(-5.1 * delta[0], 5.1 * delta[0])
    ax.set_ylim(-5.1 * delta[1], 5.1 * delta[1])

    filt_together = (p1_raw[:, scol] == p2_raw[:, scol]) & (tmp_filt)
    segs = np.where(~filt_together[:-1] == filt_together[1:])[0]
    segs = np.append(segs, len(filt_together) - 1)

    ax.plot(p1[:, 0] * conv, p1[:, 1] * conv, 'r--', alpha=0.3)
    ax.plot(p2[:, 0] * conv, p2[:, 1] * conv, 'b--', alpha=0.3)
    for tag in closest_tags[2:3]:
        tmp_path = path_lookup[tag][:, pxcol:pzcol + 1]
        tmp_path = subtract_path(tmp_path, coms)
        ax.plot(tmp_path[:, 0] * conv, tmp_path[:, 1] * conv, color='0.5')

    for tag in closest_tags_norm[2:4]:
        tmp_path = path_lookup[tag][:, pxcol:pzcol + 1]
        tmp_path = subtract_path(tmp_path, coms)
        ax.plot(tmp_path[:, 0] * conv, tmp_path[:, 1] * conv, '--', color='0.5')

    seg_last = -1
    for tmp_seg in segs:
        start = seg_last + 1
        end = tmp_seg + 1
        if filt_together[start]:
            ax.plot(p1[:, 0][start:end] * conv, p1[:, 1][start:end] * conv, 'r-', alpha=0.6)
            ax.plot(p2[:, 0][start:end] * conv, p2[:, 1][start:end] * conv, 'b-', alpha=0.6)

        seg_last = tmp_seg
    ax.annotate(r"$a_i = {0:.0f}$ au, $e_i$ = {1:.2g}".format(tmp_orb[0] * cgs.pc / cgs.au, tmp_orb[1]) + "\n" + "{0}".format(
            tmp_row), (0.01, 0.99), ha='left', va='top', xycoords='axes fraction')

    ##Getting the closest distances...maybe should look at the closest distances after the halo mass disappears...
    tag = closest_tags[2]
    tmp_path = path_lookup[tag]
    pclose = subtract_path(tmp_path[:, pxcol:pzcol + 1], coms)
    pclose = np.sum(pclose * pclose, axis=1)**.5
    tag = closest_tags[3]
    tmp_path = path_lookup[tag]
    pclose2 = subtract_path(tmp_path[:, pxcol:pzcol + 1], coms)
    pclose2 = np.sum(pclose2 * pclose2, axis=1)**.5
    tag = closest_tags[4]
    tmp_path = path_lookup[tag]
    pclose3 = subtract_path(tmp_path[:, pxcol:pzcol + 1], coms)
    pclose3 = np.sum(pclose3 * pclose3, axis=1)**.5
    tag = closest_tags[5]
    tmp_path = path_lookup[tag]
    pclose4 = subtract_path(tmp_path[:, pxcol:pzcol + 1], coms)
    pclose4 = np.sum(pclose4 * pclose4, axis=1)**.5



    ax = axs[1,0]
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")

    ax.plot(p1_raw[:, pxcol], p1_raw[:, pycol], 'r', alpha=0.5)
    ax.plot(p2_raw[:, pxcol], p2_raw[:, pycol], 'b--', alpha=0.5)
    ax.plot(p1_raw_fst[pxcol], p2_raw_fst[pycol], "rs")

    for tag in closest_tags[2:3]:
        tmp_path = path_lookup[tag]
        ax.plot(tmp_path[:, pxcol], tmp_path[:, pycol], color='0.5')

    for tag in closest_tags_norm[2:4]:
        tmp_path = path_lookup[tag]
        ax.plot(tmp_path[:, pxcol], tmp_path[:, pycol], '--', color='0.5')

    close_str1 = r"Closest distance ={0:.2g} au, time = {1:.2g} yr".format(closest_approaches[2], closest_approaches_time[2]) +\
                "\n" + "{0}\n".format(closest_tags[2])
    close_str2 = r"Min (Approach / Bin Sep) ={0:.2g}, time = {1:.2g} yr".format(closest_approaches_norm[2], closest_approaches_norm_time[2]) +\
                "\n" + "{0}\n".format(closest_tags_norm[2])
    close_str3 = "Class: {0}\n".format(tmp_class)
    close_str4 = "Initial Separation: {0:.1f}".format(psep[~np.isinf(psep)][0] * cgs.pc / cgs.au)

    ax.annotate(close_str1 + close_str2 + close_str3 + close_str4, (0.01, 0.99), ha='left', va='top', xycoords='axes fraction', fontsize=16)
    ##Could also add the time of the closest encounter...
    # fig.savefig(base + aa + "com_path_{0:03d}.pdf".format(fig_idx))

    # ax = axs[0, 1]
    # ax.set_xlim(2e6, 2e7)
    # ax.set_xlabel("Time [yr]")
    # ax.set_ylabel("Normalized tidal force")
    # ax.plot(np.linspace(0, 489, 490)[int(halo_snap[jj]) + 1:] * snap_interval, t_series[int(halo_snap[jj]) + 1:])
    # ax.annotate("Avg, Max={0:.2g}, {1:.2g}".format(t_avg, t_max),
    #             (0.01, 0.99), xycoords="axes fraction", va='top', ha='left')

    ax = axs[1,1]
    ax.set_xlim(2e6, 2e7)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Closest distances")
    # ax.semilogy(np.array([snap_max, snap_max]) * snap_interval, [0.1, 1], 'r--')
    ax.semilogy(np.linspace(0, end_snap, end_snap + 1) * snap_interval, pclose)
    ax.semilogy(np.linspace(0, end_snap, end_snap + 1) * snap_interval, pclose2)
    ax.semilogy(np.linspace(0, end_snap, end_snap + 1) * snap_interval, pclose3)
    ax.semilogy(np.linspace(0, end_snap, end_snap + 1) * snap_interval, pclose4)



    fig.savefig(base + aa + "tmp2_com_path_{0:03d}.png".format(fig_idx))
    fig_idx += 1
    sys.stdout.flush()
    print(base + aa + "tmp2_com_path_{0:03d}.png")
    del path_diff_all, mass_all, vel_all_x, vel_all_y, vel_all_z, u_tags_exclude, path_diff_all_order, p1_raw, p2_raw


    plt.clf()
np.savez(base + aa + "dens_series_bins_nrms_{0}.npz".format(nclose), dens=dens_series, hs=halo_snap,
         force=force_series, cross_sections=cross_sections, sigma=sigma_series, mean_mass=mean_mass,
         rms_mass=rms_mass)
with open(base + aa + "closest.p", "wb") as ff:
    pickle.dump(closest, ff)
###########################################################################