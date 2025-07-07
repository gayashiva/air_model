# Methodology

## 2.1 Mass Balance Model Framework

The foundation of this study builds upon the bulk energy and mass balance model developed by Balasubramanian et al. (2022) for simulating Artificial Ice Reservoir (AIR) evolution. The original model employs a cone-shaped geometry to represent AIR structures and calculates ice volume, surface temperature, and wastewater through coupled energy and mass balance equations at hourly time steps.

The mass balance equation for an AIR is represented as:

**(ΔM_F + ΔM_ppt + ΔM_dep)/Δt = (ΔM_ice + ΔM_water + ΔM_sub + ΔM_waste)/Δt**

where M_F is the cumulative mass of fountain discharge, M_ppt is cumulative precipitation, M_dep is cumulative accumulation through water vapor deposition, M_ice is cumulative mass of ice, M_water is cumulative mass of melt water, M_sub represents cumulative water vapor loss by sublimation, and M_waste represents fountain wastewater that did not interact with the AIR.

## 2.2 Spatial Extension to Mountain Grid Cells

### 2.2.1 ERA5 Data Acquisition and Processing

High-resolution ERA5 reanalysis data were downloaded for mountain grid cells across multiple countries at 0.25° spatial resolution. The dataset included hourly time series from 2001 to 2020 for the following meteorological variables:

- Air temperature (t2m) [K]
- Relative humidity derived from 2-meter dewpoint temperature (d2m) [K]  
- Wind speed components (u10, v10) [m/s]
- Global shortwave radiation (ssrd) [J/m²]

Wind speed was calculated as: **wind = √(u10² + v10²)**

Relative humidity was derived using the saturation vapor pressure relationship:
**RH = 100 × e_sat(T_dew)/e_sat(T_air)**

where e_sat is the saturation vapor pressure calculated using the empirical formula with surface-specific coefficients.

### 2.2.2 Mountain Cell Identification and Categorization

Mountain grid cells were identified using the K1 global mountains GIS datalayer (Kapos et al., 2000) with a 0.75 threshold for mountain area extent within each gridbox. The identified mountain cells were subsequently categorized based on country boundaries using spatial intersection algorithms.

## 2.3 Typical Meteorological Year Generation

### 2.3.1 TMY Algorithm Implementation

To generate representative meteorological conditions from the 20-year ERA5 dataset (2001-2020), a Typical Meteorological Year (TMY) approach was implemented. The algorithm selects the most representative month from the historical record for each calendar month by comparing statistical distributions of key meteorological variables.

For each calendar month, the algorithm evaluates all available years in the dataset and identifies which specific year provides the most representative conditions for that month. This selection is based on comparing cumulative distribution functions of meteorological variables between individual monthly records and the long-term climatological distribution for that month. The year that minimizes the statistical differences between its monthly distribution and the overall climatological distribution is selected to represent that calendar month in the typical year.

When no suitable representative month can be identified from the historical data, the algorithm falls back to using climatological averages calculated by averaging all available data for each day-of-year and hour combination across the 20-year period.

### 2.3.2 Temporal Continuity and Data Quality

The TMY generation algorithm ensured temporal continuity by:

- Creating synthetic 2-year duration datasets by duplicating and temporally shifting annual cycles
- Handling leap year inconsistencies by substituting February 29th with February 28th in non-leap years
- Interpolating missing values at month boundaries using temporal interpolation methods
- Validating data completeness and applying quality control measures

## 2.4 Model Assumptions and Parameterization

### 2.4.1 Fountain System Configuration

For each mountain grid cell simulation, standardized fountain parameters were applied:

- **Infinite discharge capacity**: 1,000,000 L/min representing unlimited water availability
- **Fountain water temperature**: Assumed to be 0°C for all simulations
- **Spray radius**: Fixed at 10 meters for all simulations
- **Initial fountain height**: Negligible value corresponding to the surface layer thickness, as dome volume was set to zero for all simulations

### 2.4.2 Meteorological Assumptions

Cloudiness effects were ignored in this study, assuming clear sky conditions for all simulations to simplify radiation calculations.

For each mountain grid cell, atmospheric pressure was estimated from altitude.

Precipitation effects on AIR evolution were not considered in this study, simplifying the mass balance calculations to focus on the primary processes of fountain water input, freezing, melting, and sublimation.

### 2.4.3 Construction Start Date Algorithm

The model initialization employs an automated start date determination algorithm based on sustained cold periods. The algorithm identifies suitable construction periods by analyzing daily minimum temperatures and counting consecutive days below the critical temperature threshold of 0°C. When the algorithm detects seven consecutive days with daily minimum temperatures below 0°C, it designates the beginning of this cold period as a potential construction start date. If temperatures rise above 0°C before completing seven consecutive days, the counter resets and begins searching for the next cold period. This approach ensures AIR construction begins only when meteorological conditions favor sustained ice formation over an extended period.

### 2.4.4 Surface Energy Balance Parameterization

The surface energy balance was calculated using:
**ρ_ice × c_ice × (ΔT/Δt) × Δx = q_SW + q_LW + q_L + q_S + q_F + q_G**

where energy fluxes include shortwave radiation (q_SW), longwave radiation (q_LW), latent heat flux (q_L), sensible heat flux (q_S), fountain heat flux (q_F), and ground heat flux (q_G).

## 2.5 Model Implementation and Execution

The extended model was implemented to enable automated processing of multiple mountain locations simultaneously. The computational workflow consisted of:

1. **Preprocessing**: ERA5 data conversion from raw netCDF format to processed CSV files with derived meteorological variables
2. **TMY Generation**: Application of the typical year algorithm to create representative annual cycles
3. **Model Execution**: Hourly time-step simulation of AIR evolution using the mass balance framework
4. **Output Processing**: Generation of ice volume time series, water efficiency metrics, and survival duration statistics

Each simulation was configured with standardized initial conditions (dome volume, surface roughness, albedo) while allowing site-specific meteorological forcing to drive model evolution. This approach enabled systematic comparison of AIR performance across diverse mountain environments while maintaining consistency in model parameterization and assumptions.

