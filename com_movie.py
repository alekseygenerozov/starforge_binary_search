import numpy as np
import matplotlib.pyplot as plt
from bash_command import bash_command as bc
import cgs_const as cgs

import sys
sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
import starforge_constants as sfc


snap_interval = 2.47e4
snap_sim_units = snap_interval / (sfc.GN ** .5 * (cgs.pc ** 3 / (cgs.G * cgs.M_sun)) ** .5 / cgs.year)

def make_com_movie(pdata, tag="", save_dir="./"):
    ##Infer the number of particles from the data.
    nparts = pdata.shape[-2]
    ##Highlight the binary
    highlight = (0, 1)
    highlight = np.array(highlight)
    cols = np.array(['0.5'] * nparts)
    cols[highlight] = "r"

    ##Initial com and separation  of the binary. Use this frame. Separation is used to set axis ranges
    snap = pdata[0]
    sep = np.linalg.norm(snap[0, 1:4] - snap[1, 1:4])
    com_highlight_start = np.average(snap[highlight, 1:7], axis=0, weights=snap[highlight, 7])
    snap = pdata[-1]
    sep = np.linalg.norm(snap[0, 1:4] - snap[1, 1:4])
    com_highlight_end = np.average(snap[highlight, 1:7], axis=0, weights=snap[highlight, 7])
    avg_vel = (com_highlight_end[:3] -  com_highlight_start[:3]) / snap_sim_units
    # offset = com_highlight[:3] + com_highlight[3:] * snap_sim_units * np.array(range(len(pdata)))
    offset = np.array([com_highlight_start[:3] + avg_vel * snap_sim_units * tt / 200. for tt in range(len(pdata))])
    # offset = np.array([com_highlight[:3] for tt in range(len(pdata))])


    for ii, snap in enumerate(pdata):
        fig,ax = plt.subplots()

        sep = np.linalg.norm(snap[0, 1:4] - snap[1, 1:4])
        com_highlight = np.average(snap[highlight, 1:7], axis=0, weights=snap[highlight, 7])
        delta = max(0.005, sep, np.linalg.norm(com_highlight[:3] - offset[ii]))
        ax.set_xlim(-delta, delta)
        ax.set_ylim(-delta, delta)

        for jj in range(nparts):
            ax.scatter(snap[jj, 1] - offset[ii, 0],
                       snap[jj, 2] - offset[ii, 1], marker="o", c=cols[jj],
                       s=snap[jj, 7] / np.max(snap[:, 7]) * 100)
            ax.plot(pdata[:ii+1, jj, 1] - offset[:ii+1,0], pdata[:ii+1, jj, 2] - offset[:ii+1,1], "-", color=cols[jj], alpha=0.5)
        fig.savefig(f"{save_dir}/movie{tag}_{ii:03d}.png")
    bc.bash_command(f"ffmpeg -r 5 -i {save_dir}/movie{tag}_%3d.png {save_dir}/move{tag}.webm")
    # bc.bash_command(f"ffmpeg -r 5-i {save_dir}/movie{tag}_%3d.png move{tag}.webm")
    # bc.bash_command(f"rm {save_dir}/movie{tag}*png")


