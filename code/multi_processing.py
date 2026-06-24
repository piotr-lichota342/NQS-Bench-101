import multiprocessing as mp
import os
from time import time
from all_architectures_control import launch_control

def worker(_):
    return launch_control()

def main():

    start_time = time()

    n_runs = 100
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    print("Running", n_runs, "tasks on", n_cores, "cores")

    with mp.Pool(n_cores) as pool:
        pool.map(worker, range(n_runs))

    print("Done in", time() - start_time)


if __name__ == "__main__":
    main()