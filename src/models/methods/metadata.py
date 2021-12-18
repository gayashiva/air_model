"""Returns Parameter metadata for web app
"""

import streamlit as st


@st.cache
def get_parameter_metadata(
    parameter,
):  # Provides Metadata of all input and Output variables
    return {
        "DX": {
            "name": "Surface layer thickness",
            "latex": "$\\Delta x$",
            "ylim": [10e-03, 100e-03],
            "step": 1e-03,
            "kind": "parameter",
            "units": "[$mm$]",
        },
        # "A_cone_corr": {
        #     "name": "Surface area correction factor",
        #     "latex": "$A_{corr}$",
        #     "ylim": [1, 2],
        #     "step": 0.1,
        #     "kind": "parameter",
        #     "units": "( )",
        # },
        "Z": {
            "name": "Surface roughness",
            "latex": "$z_{0}$",
            "ylim": [1e-03, 5e-03],
            "step": 1e-03,
            "kind": "parameter",
            "units": "($mm$)",
        },
        "R_F": {
            "name": "Spray radius",
            "latex": "$r_{F}$",
            # "ylim": [0.9, 1.1],
            "kind": "parameter",
            "units": "($m$)",
        },
        "D_F": {
            "name": "Mean Discharge",
            "latex": "$d_{F}$",
            "ylim": [0.5, 1.5],
            "kind": "parameter",
            "units": "()",
        },
        "T_F": {
            "name": "Water temperature",
            "latex": "$T_{F}$",
            "ylim": [0, 3],
            "step": 1,
            "kind": "parameter",
            "units": "($\\degree C$)",
        },
        "DT": {
            "name": "Time step",
            "latex": "$\\Delta t$",
            "kind": "parameter",
            "units": "()",
        },
        "IE": {
            "name": "Ice Emissivity",
            "latex": "$\\epsilon_{ice}$",
            "ylim": [0.95, 0.99],
            "step": 0.01,
            "kind": "parameter",
            "units": "()",
        },
        "A_DECAY": {
            "name": "Albedo decay rate",
            "latex": "$\\tau$",
            "ylim": [10, 22],
            "kind": "parameter",
            "units": "($days$)",
        },
        "A_S": {
            "name": "Snow Albedo",
            "latex": r"$\alpha_{snow}$",
            "ylim": [0.8, 0.9],
            "kind": "parameter",
            "units": "()",
        },
        "A_I": {
            "name": "Ice Albedo",
            "latex": r"$\alpha_{ice}$",
            "ylim": [0.15, 0.35],
            "step": 0.05,
            "kind": "parameter",
            "units": "()",
        },
        "T_PPT": {
            "name": "Precipitation temperature threshold",
            "latex": "$T_{ppt}$",
            "ylim": [0, 2],
            "step": 1,
            "kind": "parameter",
            "units": "($\\degree C$)",
        },
        "guttannen21": {
            "name": "Guttannen 2021",
            "shortname": "CH21",
            "slidename": "Swiss",
            "kind": "site",
            "units": "()",
        },
        "guttannen20": {
            "name": "Guttannen 2020",
            "shortname": "CH20",
            "kind": "site",
            "units": "()",
        },
        "schwarzsee19": {
            "name": "Schwarzsee 2019",
            "shortname": "CH19",
            "kind": "site",
            "units": "()",
        },
        "gangles21": {
            "name": "Gangles 2021",
            "shortname": "IN21",
            "slidename": "Indian",
            "kind": "site",
            "units": "()",
        },
        "time": {
            "name": "Timestamp",
            "kind": "Misc",
            "units": "()",
        },
        "h_s": {
            "name": "Height Steps",
            "kind": "Misc",
            "units": "()",
        },
        "cam_temp_full": {
            "name": "Camera errors",
            "kind": "Misc",
            "units": "()",
        },
        "fountain_froze": {
            "name": "Frozen Discharge",
            "kind": "Output",
            "units": "($kg\\, h^{-1}$)",
        },
        "event": {
            "name": "Freezing/Melting Event",
            "kind": "Derived",
            "units": "($kg\\, s^{-1}$)",
        },
        "h_f": {
            "name": "Fountain Height",
            "kind": "Derived",
            "units": "()",
        },
        "dia": {
            "name": "Measured diameter",
            "kind": "Misc",
            "units": "($m$)",
        },
        "cam_temp": {
            "name": "Thermal Cam Validation",
            "kind": "Derived",
            "units": "($\\degree C$)",
        },
        "DroneV": {
            "name": "Drone Validation",
            "kind": "Derived",
            "units": "($m^3$)",
        },
        "tcc": {
            "name": "Total Cloud Cover",
            "kind": "Derived",
            "units": "()",
        },
        "missing_type": {
            "name": "Column Filled from ERA5",
            "kind": "Derived",
            "units": "()",
        },
        "e_a": {
            "name": "Atmospheric Emissivity",
            "kind": "Derived",
            "units": "()",
        },
        "vp_a": {
            "name": "Air Vapour Pressure",
            "kind": "Derived",
            "units": "($hPa$)",
        },
        "vp_ice": {
            "name": "Ice Vapour Pressure",
            "kind": "Derived",
            "units": "($hPa$)",
        },
        "melted": {
            "name": "Melt per time step",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "sub": {
            "name": "Sublimation per time step",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "j_cone": {
            "name": "Thickness",
            "kind": "Derived",
            "units": "($m$)",
        },
        "Discharge": {
            "name": "Discharge",
            "kind": "Input",
            "units": "($l\\, min^{-1}$)",
        },
        "wasted": {
            "name": "Discharge Wasted",
            "kind": "Output",
            "units": "($kg\\, s^{-1}$)",
        },
        "temp": {
            "name": "Temperature",
            "kind": "Input",
            "units": "($\\degree C$)",
        },
        "delta_T_s": {
            "name": "Temperature change per time step",
            "kind": "Derived",
            "units": "($\\degree C$)",
        },
        "RH": {
            "name": "Relative Humidity",
            "kind": "Input",
            "units": "($\\%$)",
        },
        "press": {
            "name": "Pressure",
            "kind": "Input",
            "units": "($hPa$)",
        },
        "SW_global": {
            "name": "Shortwave Global",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "SW_direct": {
            "name": "Shortwave Direct",
            "kind": "Input",
            "units": "($W\\,m^{-2}$)",
        },
        "SW": {
            "name": "Net Shortwave Radiation",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "SW_diffuse": {
            "name": "Shortwave Diffuse Radiation",
            "kind": "Input",
            "units": "($W\\,m^{-2}$)",
        },
        "LW_in": {
            "name": "Incoming Longwave Radiation",
            "kind": "Input",
            "units": "($W\\,m^{-2}$)",
        },
        "LW": {
            "name": "Net Longwave Radiation",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "Qs": {
            "name": "Sensible Heat flux",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "Ql": {
            "name": "Latent Heat flux",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "Qf": {
            "name": "Fountain water heat flux",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "Qg": {
            "name": "Bulk Icestupa heat flux",
            "kind": "Derived",
            "units": "($W\\,m^{-2}$)",
        },
        "Qt": {
            "name": "Temperature flux",
            "kind": "Output",
            "units": "($W\\,m^{-2}$)",
        },
        "Qmelt": {
            "name": "Melt energy flux",
            "kind": "Output",
            "units": "($W\\,m^{-2}$)",
        },
        "Qfreeze": {
            "name": "Freeze energy flux",
            "kind": "Output",
            "units": "($W\\,m^{-2}$)",
        },
        "ppt": {
            "name": "Precipitation",
            "kind": "Input",
            "units": "($mm$)",
        },
        "wind": {
            "name": "Wind Speed",
            "kind": "Input",
            "units": "($m\\,s^{-1}$)",
        },
        "iceV": {
            "name": "Ice Volume",
            "kind": "Output",
            "units": "($m^3$)",
        },
        "ice": {
            "name": "Ice Mass",
            "kind": "Output",
            "units": "($kg$)",
        },
        "alb": {
            "name": "Albedo",
            "kind": "Derived",
            "units": "()",
        },
        "f_cone": {
            "name": "Solar Surface Area Fraction",
            "kind": "Derived",
            "units": "()",
        },
        "s_cone": {
            "name": "Ice Cone Slope",
            "kind": "Derived",
            "units": "()",
        },
        "h_cone": {
            "name": "Ice Cone Height",
            "kind": "Output",
            "units": "($m$)",
        },
        "r_cone": {
            "name": "Ice Cone Radius",
            "kind": "Output",
            "units": "($m$)",
        },
        "T_s": {
            "name": "Surface Temperature",
            "kind": "Output",
            "units": "($\\degree C$)",
        },
        "T_bulk": {
            "name": "Bulk Temperature",
            "kind": "Derived",
            "units": "($\\degree C$)",
        },
        "sea": {
            "name": "Solar Elevation Angle",
            "kind": "Derived",
            "units": "($\\degree$)",
        },
        "ghics": {
            "name": "Estimated Solar Irradiance",
            "kind": "Derived",
            "units": "($\\degree$)",
        },
        "Qtotal": {
            "name": "Net Energy",
            "kind": "Output",
            "units": "($W\\,m^{-2}$)",
        },
        "snow2ice": {
            "name": "Snow Accumulation",
            "kind": "Output",
            "units": "($kg$)",
        },
        "cdt": {
            "name": "Condensation",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "dep": {
            "name": "Deposition",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "vapour": {
            "name": "Vapour loss",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "meltwater": {
            "name": "Meltwater",
            "kind": "Output",
            "units": "($kg$)",
        },
        "wastewater": {
            "name": "Wasted Foutain Water",
            "kind": "Output",
            "units": "($kg$)",
        },
        "A_cone": {
            "name": "Surface Area",
            "kind": "Output",
            "units": "($m^2$)",
        },
        "input": {
            "name": "Mass Input",
            "kind": "Output",
            "units": "($kg$)",
        },
        "wind_loss": {
            "name": "Wind loss",
            "kind": "Derived",
            "units": "($kg$)",
        },
        "dr": {
            "name": "Radius growth rate",
            "latex": "$\\Delta y$",
            # "ylim": [10e-03, 50e-03],
            # "step": 5e-03,
            "kind": "Output",
            "units": "($m$)",
        },
    }[parameter]
