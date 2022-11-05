import os
import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    for it in ['ps', 'ts', 'snow', 'stressx', 'tstrat', 'tausurf', 'ssun', 'fuptopv', 'fuptopir', 'fupsurfir', 'fdnsurfir', 'surfalb']:

        directory = "./MY35_FG"
        numpy_dir = f"./{directory.split('_')[0]}_OUT/storm_{it}_auto"
        out_dir = f"./{directory.split('_')[0]}_IMG/{it}/"
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

            if sorted(os.listdir(directory)).index(item) not in [0, len(os.listdir(directory)) // 4, 2 * (len(os.listdir(directory)) // 4), 3 * (len(os.listdir(directory)) // 4), len(os.listdir(directory)) - 1]:
                x_vals = np.concatenate([np.linspace(0.5, 4.3, 77), np.linspace(4.35, 5.0, 14)])
                fig = plt.figure(figsize=(8, 6))
                ax = plt.gca()
                ax.set_yscale('log')
                plt.scatter(x_vals, vals, label="Simulated")
                plt.plot(x_vals, [emul_mse for _ in vals], color="red", label="Emulated")
                plt.xticks(np.arange(0.5, 5.1, .2))
                plt.xlabel("Lifted Dust Effective Radius", fontsize=15)
                plt.ylabel("Mean Squared Error (Log Domain)", fontsize=15)
                dustscenario = directory.split('_')[0].split("./")[1].split("MY")
                microns = item.split(".nc")[0]
                plt.title(f"Emulated {microns} Micron on MY {dustscenario[1]} Dust Scenario", fontsize=17)
                plt.legend()
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"{out_dir}/{microns}.jpg")
                plt.cla()

                plt.close(fig)
