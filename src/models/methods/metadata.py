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
        "guttannen21": {
            "fullname": "Guttannen 2021",
            "shortname": "CH21",
            "kind": "Misc",
            "units": "()",
        },
        "guttannen20": {
            "fullname": "Guttannen 2020",
            "shortname": "CH20",
            "kind": "Misc",
            "units": "()",
        },
        "schwarzsee19": {
            "fullname": "Schwarzsee 2019",
            "shortname": "CH19",
            "kind": "Misc",
            "units": "()",
        },
        "gangles21": {
            "fullname": "Gangles 2021",
            "shortname": "IN21",
            "kind": "Misc",
            "units": "()",
        },
        "When": {
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
        "cld": {
            "name": "Cloudiness",
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
        "fountain_froze": {
            "name": "Ice per time step",
            "kind": "Derived",
            "units": "($kg$)",
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
        "thickness": {
            "name": "Ice thickness",
            "kind": "Derived",
            "units": "($m$)",
        },
        "Discharge": {
            "name": "Discharge",
            "kind": "Input",
            "units": "($l\\, min^{-1}$)",
        },
        "fountain_runoff": {
            "name": "Discharge Runoff",
            "kind": "Output",
            "units": "($kg\\, s^{-1}$)",
        },
        "T_a": {
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
        "p_a": {
            "name": "Pressure",
            "kind": "Input",
            "units": "($hPa$)",
        },
        "SW_global": {
            "name": "Shortwave Global",
            "kind": "Misc",
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
        "Prec": {
            "name": "Precipitation",
            "kind": "Input",
            "units": "($mm$)",
        },
        "v_a": {
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
        "a": {
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
        "h_ice": {
            "name": "Ice Cone Height",
            "kind": "Output",
            "units": "($m$)",
        },
        "r_ice": {
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
        "Qsurf": {
            "name": "Net Energy",
            "kind": "Output",
            "units": "($W\\,m^{-2}$)",
        },
        "ppt": {
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
        "unfrozen_water": {
            "name": "Wasted Water Runoff",
            "kind": "Output",
            "units": "($kg$)",
        },
        "SA": {
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
    }[parameter]
