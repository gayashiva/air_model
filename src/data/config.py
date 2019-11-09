import os

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

site = input("Input the Field Site Name: ") or "guttannen"

option = (
    input("Fountain discharge option(energy, temperature, schwarzsee): ")
    or "temperature"
)

folders = dict(
    dirname=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    input_folder=os.path.join(dir, "data/interim/"),
    output_folder=os.path.join(dir, "data/processed/"),
    data_file=os.path.join(dir, "data/raw/" + site + "_aws.txt"),
    interim_folder=os.path.join(dir, "data/interim/"),
)

surface = dict(
    ie=0.96,  # Ice Emissivity ie
    we=0.95,  # Water emissivity we
    a_i=0.6,  # Albedo of Ice a_i
    a_s=0.75,  # Albedo of Snow a_s
    a_w=0.1,  # Albedo of Water a_w
    t_d=21.9,  # Albedo decay rate t_d
    z0mi=0.001,  # Ice Momentum roughness length
    z0hi=0.0001,  # Ice Scalar roughness length
)

if site == 'schwarzsee':
    fountain = dict(
        T_f=5,  # Fountain Water Temperature T_f
        d_f=0.005,  # Fountain hole diameter
        h_f=1.35,  # Fountain steps h_f
        discharge=4,  # Fountain on discharge
    )

if site == 'plaffeien':
    fountain = dict(
        T_f=5,  # Fountain Water Temperature T_f
        d_f=0.005,  # Fountain hole diameter
        h_f=1,  # Fountain steps h_f 1.35 for Schwarzsee
        discharge=4,  # Fountain on discharge in LPM
    )

if site == 'guttannen':
    fountain = dict(
        T_f=5,  # Fountain Water Temperature T_f
        d_f=0.005,  # Fountain hole diameter
        h_f=1,  # Fountain steps h_f 1.35 for Schwarzsee
        discharge=8,  # Fountain on discharge
    )

