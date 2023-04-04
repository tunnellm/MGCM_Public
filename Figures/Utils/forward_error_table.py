import numpy as np
from h5py import File
import os


if __name__ == "__main__":

    it = "ts"

    directory = "./MY24_FG"
    numpy_dir = f"{directory.split('_')[0]}_OUT/storm_{it}_redo_4"

    for item in sorted(os.listdir(directory)):
        numpy_loaded = np.load(f"{numpy_dir}/{'.'.join(item.split('.')[:-1])}.npy")

        with File(f"{directory}/{item}", 'r') as f_2:
            #test_item = np.average(np.average(np.average((numpy_loaded - np.array(f_2[it])) ** 2, axis=0), axis=0), axis=0)
            test_item = np.average(np.reshape((numpy_loaded - np.array(f_2[it])) ** 2, newshape=(-1, 60)), axis=0)
            print(".".join(item.split('.')[:-1]) + ',', ",".join([str(x) for x in test_item]))
