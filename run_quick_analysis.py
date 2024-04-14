import sys
import subprocess
import glob

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]


snap_base = "/scratch3/03532/mgrudic/STARFORGE_RT/STARFORGE_v1.1/" + sys.argv[1] + "/output/snapshot"

snaps = glob.glob(snap_base + "*.hdf5")
snaps = [int(ss.replace(snap_base, "").replace("_", "").replace(".hdf5", "")) for ss in snaps]
start_snap = min(snaps)
end_snap = max(snaps)

bash_command("mkdir {0}".format(sys.argv[1]))
bash_command("python3 find_multiples_new2.py --name_tag {0} --snap_base {1} {2} --tides_factor {3} ".format(sys.argv[1], snap_base, end_snap, 8.0))
# bash_command("python3 stellar_tides.py {0} {1} {2} {3} ".format(sys.argv[1], snap_base, start_snap, end_snap))


