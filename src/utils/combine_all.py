"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys
import logging, coloredlogs
import xarray as xr

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        # level=logging.WARNING,
        level=logging.INFO,
        logger=logger,
    )

    locations = ["gangles21", "guttannen21", "guttannen20"]
    # icestupas = [Icestupa(location) for location in locations]
    icestupas = []

    for location in locations:
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.df["location"] = location
        ds = icestupa.df.rename(columns={"TIMESTAMP": "time"})
        ds = ds.set_index(["time", "location"])
        ds = ds.to_xarray()
        icestupas.append(ds)

    # times = pd.date_range("2000-01-01", periods=4)
    # ds = xr.DataArray(icestupas)
    # print(ds)
    # all = xr.combine_by_coords(icestupas, compat="override")
    all = xr.combine_by_coords(icestupas)
    print(all)
    all.to_netcdf("data/all_results.nc")
