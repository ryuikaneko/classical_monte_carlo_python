
Build:
  make -f Makefile_kagome_rex_mpi

Run example:
  mpirun -np 8 ./kagome_rex_mpi --L 6 --nrep 28 --Tmin 1e-3 --Tmax 0.5 --therm 2000 --meas 5000 --sample_every 10 --samples-per-block 50 --out kagome_L6
  mpirun -np 9 ./kagome_rex_mpi --L 6 --nrep 36 --Tmin 1e-3 --Tmax 0.5 --therm 100000 --meas 5000000 --sample_every 10 --samples-per-block 50 --out kagome_L6

Debug lattice example:
  mpirun -np 1 ./kagome_rex_mpi --L 6 --nrep 28 --therm 0 --meas 0 --debug-lattice

Notes:
- CSV output columns are now:
    T, E_per_spin, C_per_spin, C_per_spin_err, count, used_blocks
- C_per_spin_err is computed by delete-one-block jackknife using block-wise accumulators of E and E^2.
- Block assignment is based on measurement sample index, while energies are sorted by the instantaneous temperature labels rep.T.
- Use --samples-per-block to tune the block size for the heat-capacity error bar.
