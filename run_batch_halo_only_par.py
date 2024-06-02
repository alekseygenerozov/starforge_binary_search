import sys
import subprocess
import glob

def bash_command(cmd, **kwargs):
	'''Run command from the bash shell'''
	process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
	return process.communicate()[0]


with open("data_loc", "r") as ff:
	snap_base = ff.read()
snaps = glob.glob(snap_base + "*hdf5")
##FOR TESTING
# snap_base = "snapshot"
start = int(sys.argv[1])
end = len(snaps)
if len(sys.argv[2]) > 2:
	end = sys.argv[2]
print(start, end)
for ii in range(start, end, 1):
	bash_command("python3 halo_masses_single_double_par.py --non_pair --tides_factor 8.0 --snap_base {0}  {1}".format(snap_base, ii))
	# bash_command("python3 halo_masses_single_double_par.py --non_pair --tides_factor 8.0 --compress --snap_base {0}  {1}".format(snap_base, ii))
	# bash_command("python3 halo_masses_single_double_par.py --non_pair --tides_factor 1.0 --snap_base {0}  {1}".format(snap_base, ii))
