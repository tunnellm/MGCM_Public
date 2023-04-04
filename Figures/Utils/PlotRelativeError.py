import os

import matplotlib.pyplot as plt
from h5py import File
import numpy as np

out_dir = f"./fig_out/"

simulator_output_file = "./1.00.nc"
emulator_output_file = "./1.00.npy"
target_output = "ts"

with File(simulator_output_file, 'r') as f:
    latitudes = np.array(f["lat"])
    longitudes = np.array(f["lon"])
    sols = np.array(f["areo"])

with File(simulator_output_file) as f:
    simulated = np.array(f[target_output])

emulated = np.load(emulator_output_file)

min_temp = 120
max_temp = 311
steps = 400
bin = 7

for long in [42]:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    cf1 = ax1.contourf(
        np.arange(0, 140) * 5,
        latitudes,
        np.swapaxes(np.reshape(emulated[:, bin, :, long], (-1, 36)), axis2=0, axis1=1),
        np.linspace(min_temp, max_temp, steps),
        cmap="jet"
    )
    cf2err = np.abs((np.swapaxes(np.reshape(simulated[:, bin, :, long], (-1, 36)), axis2=0, axis1=1) -
                     np.swapaxes(np.reshape(emulated[:, bin, :, long], (-1, 36)), axis2=0, axis1=1)) /
                    np.swapaxes(np.reshape(simulated[:, bin, :, long], (-1, 36)), axis2=0, axis1=1)) * 100

    cflin = np.arange(0,
                      np.max(np.max(cf2err)).astype(np.int32) + 2,
                      1)

    err1 = ax3.contourf(
        np.arange(0, 140) * 5,
        latitudes,
        cf2err,
        cflin,
        cmap='gray_r'
    )

    cf2 = ax2.contourf(
        np.arange(0, 140) * 5,
        latitudes,
        np.swapaxes(np.reshape(simulated[:, bin, :, long], (-1, 36)), axis2=0, axis1=1),
        np.linspace(min_temp, max_temp, steps),
        cmap='jet'
    )

    ax1.set_title(f"Gaussian Process Emulated")
    ax2.set_title(f"Global Climate Model Output")
    ax3.set_title(f"Emulated Error Vs. Simulation")
    ax3.set_xlabel("Sols")

    plt.setp((ax1, ax2, ax3), ylabel="Latitude")
    plt.suptitle(f"Comparison of Emulated Vs. Simulated Temperature With Dust Particle Size 1.0 Microns\n", fontsize=16)

    fig.tight_layout()

    plt.colorbar(cf1, ax=ax1, label='Temperature K$^\circ$')
    plt.colorbar(cf2, ax=ax2, label='Temperature K$^\circ$')
    plt.colorbar(err1, ax=ax3, label='Error: Temperature K$^\circ$')

    os.makedirs(f"./{out_dir}/", exist_ok=True)
    plt.savefig(f"./{out_dir}/forward_error_{longitudes[long]}.png",
                format="png", dpi=1200)

    plt.cla()
    # plt.show()
