"""Returns Parameter metadata for web app
"""

import streamlit as st

# from redis_cache import cache_it

# @cache_it(limit=1000, expire=None)
@st.cache
def get_parameter_metadata(
    parameter,
):  # Provides Metadata of all input and Output variables
    return {
        "DX": {
            "long_name": "Surface layer thickness",
            "latex": "$\\Delta x$",
            "ylim": [10e-03, 50e-03],
            "step": 5e-03,
            "kind": "parameter",
            "units": "mm",
        },
        "SA_corr": {
            "long_name": "Surface area correction factor",
            "latex": "$A_{corr}$",
            "ylim": [1, 2],
            "step": 0.1,
            "kind": "parameter",
            "units": "",
        },
        "Z": {
            "long_name": "Surface roughness",
            "latex": "$z_{0}$",
            "ylim": [1e-03, 5e-03],
            "step": 1e-03,
            "kind": "parameter",
            "units": "mm",
        },
        "R_F": {
            "long_name": "Spray radius",
            "latex": "$r_{F}$",
            "ylim": [0.9, 1.1],
            "kind": "parameter",
            "units": "m",
        },
        "D_F": {
            "long_name": "Mean Discharge",
            "latex": "$d_{mean}$",
            "ylim": [0.5, 1.5],
            "kind": "parameter",
            "units": "",
        },
        "T_F": {
            "long_name": "Water temperature",
            "latex": "$T_{water}$",
            "ylim": [0, 3],
            "step": 1,
            "kind": "parameter",
            # "units": "($\\degree C$)",
            "units": "degree_C",
        },
        "DT": {
            "long_name": "Time step",
            "latex": "$\\Delta t$",
            "kind": "parameter",
            "units": "",
        },
        "IE": {
            "long_name": "Ice Emissivity",
            "latex": "$\\epsilon_{ice}$",
            "ylim": [0.95, 0.99],
            "step": 0.01,
            "kind": "parameter",
            "units": "",
        },
        "A_DECAY": {
            "long_name": "Albedo decay rate",
            "latex": "$\\tau$",
            "ylim": [10, 22],
            "kind": "parameter",
            "units": "days",
        },
        "A_S": {
            "long_name": "Snow Albedo",
            "latex": r"$\alpha_{snow}$",
            "ylim": [0.8, 0.9],
            "kind": "parameter",
            "units": "",
        },
        "A_I": {
            "long_name": "Ice Albedo",
            "latex": r"$\alpha_{ice}$",
            "ylim": [0.15, 0.35],
            "step": 0.05,
            "kind": "parameter",
            "units": "",
        },
        "T_PPT": {
            "long_name": "Precipitation temperature threshold",
            "latex": "$T_{ppt}$",
            "ylim": [0, 2],
            "step": 1,
            "kind": "parameter",
            "units": "degree_C",
        },
        "guttannen21": {
            "long_name": "Guttannen 2021",
            "shortname": "CH21",
            "kind": "site",
            "units": "",
        },
        "guttannen20": {
            "long_name": "Guttannen 2020",
            "shortname": "CH20",
            "kind": "site",
            "units": "",
        },
        "schwarzsee19": {
            "long_name": "Schwarzsee 2019",
            "shortname": "CH19",
            "kind": "site",
            "units": "",
        },
        "gangles21": {
            "long_name": "Gangles 2021",
            "shortname": "IN21",
            "kind": "site",
            "units": "",
        },
        "time": {
            "long_name": "Timestamp",
            "standard_name": "Timestamp",
            "kind": "Misc",
            "units": "",
        },
        "h_s": {
            "long_name": "Height Steps",
            "kind": "Misc",
            "units": "",
        },
        "cam_temp_full": {
            "long_name": "Camera errors",
            "kind": "Misc",
            "units": "",
        },
        "fountain_froze": {
            "long_name": "Frozen Discharge",
            "kind": "Output",
            "units": "kg s-1",
        },
        "event": {
            "long_name": "Freezing/Melting Event",
            "kind": "Derived",
            "units": "kg s-1",
        },
        "h_f": {
            "long_name": "Fountain Height",
            "kind": "Derived",
            "units": "",
        },
        "dia": {
            "long_name": "Measured diameter",
            "kind": "Misc",
            "units": "m",
        },
        "cam_temp": {
            "long_name": "Thermal Cam Validation",
            "kind": "Derived",
            "units": "degree_C",
        },
        "DroneV": {
            "long_name": "Drone Validation",
            "kind": "Derived",
            "units": "m3",
        },
        "cld": {
            "long_name": "Cloudiness",
            "kind": "Derived",
            "units": "",
        },
        "missing_type": {
            "long_name": "Column Filled from ERA5",
            "kind": "Derived",
            "units": "",
        },
        "e_a": {
            "long_name": "Atmospheric Emissivity",
            "kind": "Derived",
            "units": "",
        },
        "vp_a": {
            "long_name": "Air Vapour Pressure",
            "kind": "Derived",
            "units": "hPa",
        },
        "vp_ice": {
            "long_name": "Ice Vapour Pressure",
            "kind": "Derived",
            "units": "hPa",
        },
        "melted": {
            "long_name": "Melt per time step",
            "kind": "Derived",
            "units": "kg",
        },
        "sub": {
            "long_name": "Sublimation per time step",
            "kind": "Derived",
            "units": "kg",
        },
        "t_cone": {
            "long_name": "Thickness",
            "kind": "Derived",
            "units": "m",
        },
        "Discharge": {
            "long_name": "Discharge",
            "kind": "Input",
            "units": "l min-1",
        },
        "fountain_runoff": {
            "long_name": "Discharge Runoff",
            "kind": "Output",
            "units": "kg s-1",
        },
        "T_A": {
            "long_name": "Temperature",
            "standard_name": "air_temperature",
            "kind": "Input",
            "units": "degree_C",
        },
        "delta_T_s": {
            "long_name": "Temperature change per time step",
            "kind": "Derived",
            "units": "degree_C",
        },
        "rh": {
            "long_name": "Relative Humidity",
            "standard_name": "relative_humidity",
            "kind": "Input",
            "units": "%",
        },
        "press": {
            "long_name": "Pressure",
            "standard_name": "air_pressure",
            "kind": "Input",
            "units": "hPa",
        },
        # "SW_global": {
        #     "long_name": "Shortwave Global",
        #     "kind": "Misc",
        #     "units": "W m-2",
        # },
        "SW_direct": {
            "long_name": "Shortwave Direct",
            "standard_name": "direct_downwelling_shortwave_flux_in_air",
            "kind": "Input",
            "units": "W m-2",
        },
        "SW": {
            "long_name": "Net Shortwave Radiation",
            "standard_name": "downwelling_shortwave_flux_in_air",
            "kind": "Derived",
            "units": "W m-2",
        },
        "SW_diffuse": {
            "long_name": "Shortwave Diffuse Radiation",
            "standard_name": "diffuse_downwelling_shortwave_flux_in_air",
            "kind": "Input",
            "units": "W m-2",
        },
        "LW_in": {
            "long_name": "Incoming Longwave Radiation",
            "kind": "Input",
            "units": "W m-2",
        },
        "LW": {
            "long_name": "Net Longwave Radiation",
            "kind": "Derived",
            "units": "W m-2",
        },
        "Qs": {
            "long_name": "Sensible Heat flux",
            "kind": "Derived",
            "units": "W m-2",
        },
        "Ql": {
            "long_name": "Latent Heat flux",
            "kind": "Derived",
            "units": "W m-2",
        },
        "Qf": {
            "long_name": "Fountain water heat flux",
            "kind": "Derived",
            "units": "W m-2",
        },
        "Qg": {
            "long_name": "Bulk Icestupa heat flux",
            "kind": "Derived",
            # "units": "(W\\,m^{-2}$)",
            "units": "W m-2",
        },
        "Qt": {
            "long_name": "Temperature flux",
            "kind": "Output",
            # "units": "($W\\,m^{-2}$)",
            "units": "W m-2",
        },
        "Qmelt": {
            "long_name": "Melt energy flux",
            "kind": "Output",
            # "units": "($W\\,m^{-2}$)",
            "units": "W m-2",
        },
        "Qfreeze": {
            "long_name": "Freeze energy flux",
            "kind": "Output",
            # "units": "($W\\,m^{-2}$)",
            "units": "W m-2",
        },
        "PRECIP": {
            "long_name": "Precipitation",
            "kind": "Input",
            "units": "mm",
        },
        "WS": {
            "long_name": "Wind Speed",
            "kind": "Input",
            "units": "m s-1",
        },
        "iceV": {
            "long_name": "Ice Volume",
            "kind": "Output",
            "units": "m3",
        },
        "ice": {
            "long_name": "Ice Mass",
            "kind": "Output",
            "units": "kg",
        },
        "a": {
            "long_name": "Albedo",
            "kind": "Derived",
            "units": "",
        },
        "f_cone": {
            "long_name": "Solar Surface Area Fraction",
            "kind": "Derived",
            "units": "",
        },
        "s_cone": {
            "long_name": "Ice Cone Slope",
            "kind": "Derived",
            "units": "",
        },
        "h_ice": {
            "long_name": "Ice Cone Height",
            "kind": "Output",
            "units": "m",
        },
        "r_ice": {
            "long_name": "Ice Cone Radius",
            "kind": "Output",
            "units": "m",
        },
        "T_s": {
            "long_name": "Surface Temperature",
            "kind": "Output",
            "units": "degree_C",
        },
        "T_bulk": {
            "long_name": "Bulk Temperature",
            "kind": "Derived",
            "units": "degree_C",
        },
        "sea": {
            "long_name": "Solar Elevation Angle",
            "kind": "Derived",
            "units": "degree",
        },
        "Qsurf": {
            "long_name": "Net Energy",
            "kind": "Output",
            "units": "W m-2",
        },
        "ppt": {
            "long_name": "Snow Accumulation",
            "kind": "Output",
            "units": "kg",
        },
        "cdt": {
            "long_name": "Condensation",
            "kind": "Derived",
            "units": "kg",
        },
        "dep": {
            "long_name": "Deposition",
            "kind": "Derived",
            "units": "kg",
        },
        "vapour": {
            "long_name": "Vapour loss",
            "kind": "Derived",
            "units": "kg",
        },
        "meltwater": {
            "long_name": "Meltwater",
            "kind": "Output",
            # "units": "($kg$)",
            "units": "kg",
        },
        "unfrozen_water": {
            "long_name": "Wasted Water Runoff",
            "kind": "Output",
            # "units": "($kg$)",
            "units": "kg",
        },
        "SA": {
            "long_name": "Surface Area",
            "kind": "Output",
            "units": "m2",
        },
        "input": {
            "long_name": "Mass Input",
            "kind": "Output",
            "units": "kg",
        },
        "wind_loss": {
            "long_name": "Wind loss",
            "kind": "Derived",
            "units": "kg",
        },
    }[parameter]
