from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from numba import jit
from numba.typed import List
from h5py import File
import numpy as np
import time as get_time
from time import sleep
import os

"""
@Author: Marc Tunnell
Emulate.py: Fits the emulator to the MGCM model output, then performs the emulation. Emulator output is
    then saved to disk as NumPy files.
    
This file should be placed in the root directory of the NASA GCM installation being used.
"""


def main():
    for target in ['ts', 'ps']:

        # The range of data that we would like to emulate. range_start should be NO LESS than the lowest
        #   value of the training data. range_stop should be NO GREATER than the highest value of the
        #   training data. The values of range_start and range_stop are inclusive.
        #
        # The following setup assumes a training set between 0.5 and 5.0, and emulates the input
        #   with a step size of 0.05. This would emulate the values 0.5, 0.55, ..., 4.95, 5.0
        range_start = 0.5
        range_stop = 5.0
        step_size = 0.05
        range_of_interest = np.arange(range_start, range_stop + step_size, step_size)

        # Directory of netCDF files to be used
        directory = "./MY24_FG"

        # This is used to create the folder where the output is stored.
        # By default, this will create a folder called ./MY24_OUT/ts/
        output_name = f"{directory.split('_')[0]}_OUT/{target}"

        # Gathers the names of the items in the training set directory.
        training_data_input = sorted(os.listdir(directory))

        #       *************************************************************************************************************
        #                                    Do not modify code within these asterisks
        def training_point_indices(num_points):
            i = [0, ]
            for n in range(1, num_points - 1):
                i.append(n * (len(training_data_input) // num_points))
            i.append(len(training_data_input) - 1)
            return i

        with File(f"{directory}/{training_data_input[0]}", 'r') as f:
            details = np.array(f[target])
        time_length, bins_length, lats_length, long_length = details.shape[0], details.shape[1], \
                                                             details.shape[2], details.shape[3]

        mod = np.array([training_data_input[i].split(".nc")[0] for i in items])
        mod_float = np.array(mod).astype(np.float32)

        print("Training Items:", mod)

        start_time = get_time.time()
        #       *************************************************************************************************************

        # The indices that will be used to train on in training_data_input.
        #   We recommend some number of equidistant points, and the training_point_indices function
        #   will handle this for you as shown by default in this file. If using the entire contents of
        #   the directory, this will be equivalent to items = range(len(training_data_input))
        items = training_point_indices(5)

        # These can be modified to split along different dimensions. The code will handle splitting
        # along any dimension. We found the following to be optimal in the majority of
        # situations but your experience may differ. Each column length must be divided evenly
        time_split = time_length  # column 1
        bins_split = bins_length  # column 2
        lats_split = lats_length  # column 3
        long_split = 1  # column 4

        #       *************************************************************************************************************
        #                                       Nothing below this should be modified.
        #       *************************************************************************************************************

        @jit(nopython=True)
        def load(files, inp):
            y_vals = []
            for a, b, c, d, e in inp:
                y_vals.append((files[np.where(mod_float == a)[0][0]])[int(b)][int(c)][int(d)][int(e)])
            return y_vals

        gp_dims = len(items)

        # Create a list of used column channels for training and inference.
        # This is created automatically based on the dimension splits
        # that were chosen above.
        chann_items = [
            (time_split, time_length),
            (bins_split, bins_length),
            (lats_split, lats_length),
            (long_split, long_length)]

        used_channels = [0, ]
        for chann, items in enumerate(chann_items):
            if items[0] != items[1]:
                used_channels.append(chann + 1)

        model_time = np.linspace(0, time_length - 1, time_length)
        model_bins = np.linspace(0, bins_length - 1, bins_length)
        model_lats = np.linspace(0, lats_length - 1, lats_length)
        model_long = np.linspace(0, long_length - 1, long_length)

        time_split_array = np.linspace(0, time_length, time_split + 1).astype(np.int32)
        bins_split_array = np.linspace(0, bins_length, bins_split + 1).astype(np.int32)
        lats_split_array = np.linspace(0, lats_length, lats_split + 1).astype(np.int32)
        long_split_array = np.linspace(0, long_length, long_split + 1).astype(np.int32)

        # Ensure that splits divide model output indices
        assert model_time.shape[0] % time_split == 0
        assert model_bins.shape[0] % bins_split == 0
        assert model_long.shape[0] % long_split == 0
        assert model_lats.shape[0] % lats_split == 0
        x_array = np.empty((time_length * bins_length * lats_length * long_length * len(mod), gp_dims),
                           dtype=np.float32)

        rep = np.reshape(np.tile(np.array(mod).astype(np.float32),
                                 int(x_array.shape[0] / (
                                         len(mod) * time_split * bins_split * lats_split * long_split))),
                         (-1, 1))

        # Creates the training data set from the indices given earlier in the code.
        @jit(nopython=True)
        def make_x(x_in, m_in, tiled_m):

            def make_sub_array(input_array, input_i):
                s = np.reshape(np.repeat(
                    np.linspace(input_array[input_i], input_array[input_i + 1] - 1, input_array[1]), len(m_in)),
                    (-1, 1))
                d = np.copy(s)
                for _ in range(int((x_in.shape[-2] / input_array[1] - 1) / len(m_in))):
                    d = np.vstack((d, s))
                return np.reshape(d, (-1, 1))

            x_in = np.reshape(x_in, (time_split, bins_split, lats_split, long_split, -1, gp_dims))
            for ti in range(time_split):
                for bi in range(bins_split):
                    for la in range(lats_split):
                        for lo in range(long_split):
                            time_ = make_sub_array(time_split_array, ti)
                            bins_ = make_sub_array(bins_split_array, bi)
                            lats_ = np.reshape(np.repeat(
                                np.linspace(lats_split_array[la], lats_split_array[la + 1] - 1, lats_split_array[1]),
                                int(x_in.shape[-2] / lats_split_array[1])), (-1, 1))
                            long_ = make_sub_array(long_split_array, lo)
                            x_in[ti, bi, la, lo] = np.hstack((tiled_m, time_, bins_, lats_, long_))
            return x_in

        x = np.reshape(make_x(x_array, mod, rep), (-1, gp_dims))

        infer = np.copy(x[::len(mod), ])

        # Validate the worker split
        print(f"Number of Workers: {time_split * long_split * lats_split * bins_split}")
        print(f"Elements / Worker: {x.shape[0] / (time_split * long_split * lats_split * bins_split)}")
        print(f"Total Work: {x.shape[0]}")
        sleep(2)

        # Load all NumPy training files and place in a format
        # that can be interpreted by numba.
        loaded = [np.array(File(f"{directory}/{str(file).zfill(1)}.nc", 'r')[target]) for file in mod]
        typed_loaded = List()
        [typed_loaded.append(q) for q in loaded]

        y = np.array(load(typed_loaded, x))

        time_to_stack = get_time.time()
        # Reshape the x, y arrays based on the split.
        x = np.reshape(x, (time_split, bins_split, lats_split, long_split, -1, gp_dims))
        y = np.reshape(y, (time_split, bins_split, lats_split, long_split, -1))
        infer = np.reshape(infer, (time_split, bins_split, lats_split, long_split, -1, gp_dims))
        workers = np.empty((time_split, bins_split, lats_split, long_split), dtype=object)

        kern = RationalQuadratic()
        # Fit the emulator to the training data. The remainder of the code from this point should be trivial
        # to parallelize. Some strange behavior occurred with the Python Multiprocessing Library and so this
        # has been left to future work.
        for t in range(time_split):
            print("Training:", t + 1, '/', time_split)
            for b in range(bins_split):
                print("Bins:", b + 1, '/', bins_split)
                for o in range(lats_split):
                    for n in range(long_split):
                        workers[t, b, o, n] = GaussianProcessRegressor(kernel=kern, normalize_y=True).fit(
                            x[t, b, o, n][:, used_channels], y[t, b, o, n])

        fitted_model_time = get_time.time()
        print(fitted_model_time - start_time)

        # Performs the emulation and saves the output
        phph = np.full((time_length, bins_length, lats_length, long_length), 3.)
        for i in enumerate(range_of_interest):
            infer[:, :, :, :, :, 0] = float(i[1])
            print("Working on:", i)
            for t in range(time_split):
                for b in range(bins_split):
                    for o in range(lats_split):
                        for n in range(long_split):
                            estimate = workers[t, b, o, n].predict(
                                infer[t, b, o, n][:, used_channels], )
                            phph[time_split_array[t]:time_split_array[t + 1],
                            bins_split_array[b]:bins_split_array[b + 1],
                            lats_split_array[o]:lats_split_array[o + 1],
                            long_split_array[n]:long_split_array[n + 1]] = np.reshape(estimate,
                                                                                      (time_split_array[1],
                                                                                       bins_split_array[1],
                                                                                       lats_split_array[1],
                                                                                       long_split_array[1]))
            os.makedirs(output_name, exist_ok=True)
            np.save(f"./{output_name}/{i[1]}", phph)

        done_time = get_time.time()

        print("Time to fit model:", '%s' % (fitted_model_time - start_time))
        print("Time to stack data:", '%s' % (time_to_stack - start_time))
        print("Time to process estimate:", '%s' % (done_time - fitted_model_time))
        print("Total Time:", '%s' % (done_time - start_time))


if __name__ == '__main__':
    main()
    exit()
