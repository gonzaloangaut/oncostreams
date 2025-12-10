from tumorsphere.core.simulation import Simulation
from tumorsphere.core.forces import (
    No_Forces,
    Spring_Force,
    Vicsek,
    Vicsek_and_Spring_Force,
    Grosmann,
    Anisotropic_Grosmann,
)
import numpy as np
import time

start = time.time()

# Create force
# force = Anisotropic_Grosmann(
#     kRep=10/3,
#     bExp=3,
#     noise_eta=0.1/3,
#     shrinking= False,
# )

force_1 = Anisotropic_Grosmann(
    kRep=1,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)

force_2 = Anisotropic_Grosmann(
    kRep=5,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)

force_3 = Anisotropic_Grosmann(
    kRep=10,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)

force_4 = Anisotropic_Grosmann(
    kRep=15,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)

force_5 = Anisotropic_Grosmann(
    kRep=20,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)

force_6 = Anisotropic_Grosmann(
    kRep=30,
    bExp=3,
    D_par=0.01,
    D_perp=0.01,
    D_phi=0.01,
    shrinking= False,
)


aspect_ratio_max = 1.01
# Defino la simulación
sim = Simulation(
    forces=[force_1],#[force_1, force_2, force_3, force_4, force_5, force_6],
    num_of_realizations=1,
    num_of_steps_per_realization=5_000, #50_000,
    initial_number_of_cells=[1_000],
    #initial_fraction_elongated=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    initial_fraction_elongated=[1.0],
    initial_density=[0.5],
    reproduction=False,
    movement=True,
    deformation=False,
    overlap_threshold_ratio=0.35,
    grid_cube_size=2*np.sqrt(aspect_ratio_max),
    delta_t=0.05,
    aspect_ratio_max=aspect_ratio_max, 
    delta_aspect_ratio=0.1,
)

# Ahora simulamos
sim.simulate_parallel(
    sql=False,
    dat_files=False,
    dat_pos_ar=True,
    save_step_dat_pos_ar=1000,
    dat_order_par=True,
    save_step_dat_order_par=1000,
    ovito=True,
    save_step_ovito=1000,
    number_of_processes=10,
    output_dir="ejemplito_ovito",
)


end = time.time()
print(f"Tiempo de ejecución: {end - start:.4f} segundos")
