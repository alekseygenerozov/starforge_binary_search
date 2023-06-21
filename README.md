# starforge_binary_search

Idenfies all of the binaries in a starforge simulation snapshot.

```
usage: find_multiples_new2.py [-h] [--snap_base SNAP_BASE] [--sma_order]
                              [--halo_mass_file HALO_MASS_FILE]
                              [--mult_max MULT_MAX] [--ngrid NGRID]
                              [--compress] [--tides_factor TIDES_FACTOR]
                              snap

Parse starforge snapshot, and get multiple data.

positional arguments:
  snap                  Snapshot index

optional arguments:
  -h, --help            show this help message and exit
  --snap_base SNAP_BASE
                        First part of snapshot name
  --sma_order           Assemble hierarchy by sma instead of binding energy
  --halo_mass_file HALO_MASS_FILE
                        Name of file containing gas halo mass around sink
                        particles
  --mult_max MULT_MAX   Multiplicity cut (4).
  --ngrid NGRID         Number of subgrids to use. Higher number will be
                        faster, but less accurate (1)
  --compress            Filter out compressive tidal forces
  --tides_factor TIDES_FACTOR
                        Prefactor for check of tidal criterion (8.0)

  ```
  
  Output: Two pickle files with all of the binary data stored. One pickle file (with TidesFalse) has the multiples without any cuts from tidal forces. 
  The other one includes the cut from tidal forces. 
 
  
  

