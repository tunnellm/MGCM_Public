import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    for it in ['ps', 'ts', 'snow', 'stressx', 'tstrat', 'tausurf', 'ssun', 'fuptopv', 'fuptopir', 'fupsurfir', 'fdnsurfir', 'surfalb']:
        vals_yep = []
        x_yep = []
        directory = "./MY24_FG"
        numpy_dir = f"./{directory.split('_')[0]}_OUT/storm_{it}_auto"
        out_dir = f"./{directory.split('_')[0]}_IMG_HOL/{it}/"
        os.makedirs(out_dir, exist_ok=True)
        for item in sorted(os.listdir(directory)):
            with File(f"{directory}/{item}", 'r') as f:
                test_item = np.reshape(np.array(f[it]), (-1))

            # test_item = np.reshape(np.load(f"{numpy_dir}/{item}"), (-1))
            print(f"\nBase Array: {item}")
            vals = []
            for comparison in sorted(os.listdir(directory)):
                if sorted(os.listdir(directory)).index(item) in [0, len(os.listdir(directory)) // 4, 2 * (len(os.listdir(directory)) // 4), 3 * (len(os.listdir(directory)) // 4), len(os.listdir(directory)) - 1]:
                    break
                if item.split(".npy")[0] != comparison.split(".nc")[0]:
                    with File(f"{directory}/{comparison}", 'r') as f_2:
                        co = np.reshape(np.array(f_2[it]), (-1))
                        mse = mean_squared_error(co, test_item)
                        emul_mse = mean_squared_error(test_item, np.reshape(np.load(f"{numpy_dir}/{sorted(os.listdir(numpy_dir))[sorted(os.listdir(directory)).index(item)]}"), (-1)))
                        vals.append(mse)
                        print(mse, end=", ")
                else:
                    print("np.NaN", end=", ")
                    vals.append(np.NaN)
            x_yep.append(float(item.split(".nc")[0]))
            if sorted(os.listdir(directory)).index(item) not in [0, len(os.listdir(directory)) // 4, 2 * (len(os.listdir(directory)) // 4), 3 * (len(os.listdir(directory)) // 4), len(os.listdir(directory)) - 1]:
                where = np.where(np.logical_and(np.array(vals) < emul_mse, np.array(vals) > 0))[0]
                app = np.abs((len(x_yep) - 1) - where) if len(where) > 0 else np.nan
                vals_yep.append(np.min(app).astype(np.float32))
            else:
                vals_yep.append(np.nan)

        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        plt.scatter(x_yep, np.array(vals_yep) * .05, label="Error")
        plt.xticks(np.arange(0.5, 5.1, .2), rotation=35)
        plt.xlabel("GCM Parameter Parameter Values")
        plt.ylabel("Delta")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{item}.jpg")
        plt.cla()
        plt.close(fig)

