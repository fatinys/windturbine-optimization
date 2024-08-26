# Data Dictionary


## Turbine Data
    
| Key           | Value Type      | Key Description |
|---------------|-----------------|-----------------|
| case_id       | number (integer) | Unique stable identification number. |
| faa_ors       | string           | Unique identifier for cross-reference to the Federal Aviation Administration (FAA) digital obstacle files. |
| faa_asn       | string           | Unique identifier for cross-reference to the FAA obstruction evaluation airport airspace analysis dataset. |
| usgs_pr_id    | number (integer) | Unique identifier for cross-reference to the 2014 USGS turbine dataset. |
| t_state       | string           | State where turbine is located. |
| t_county      | string           | County where turbine is located. |
| t_fips        | string           | State and county FIPS where turbine is located, based on spatial join of turbine points with US state and county. |
| p_name        | string           | Name of the wind power project that the turbine is a part of. Project names are typically provided by the developer; some names are identified via other internet resources, and others are created by the authors to differentiate them from previous projects. Values that were unknown were assigned a name based on the county where the turbine is located. |
| p_year        | number (integer) | Year that the turbine became operational and began providing power. Note this may differ from the year that construction began. |
| p_tnum        | number (integer) | Number of turbines in the wind power project. |
| p_cap         | number (float)   | Cumulative capacity of all turbines in the wind power project in megawatts (MW). |
| t_manu        | string           | Turbine manufacturer - name of the original equipment manufacturer of the turbine. |
| t_model       | string           | Turbine model - manufacturer's model name of each turbine. |
| t_cap         | number (integer) | Turbine rated capacity - stated output power at rated wind speed from manufacturer, ACP, and/or internet resources in kilowatts (kW). |
| t_hh          | number (float)   | Turbine hub height in meters (m). |
| t_rd          | number (float)   | Turbine rotor diameter in meters (m). |
| t_rsa         | number (float)   | Turbine rotor swept area in square meters (m²). |
| t_ttlh        | number (float)   | Turbine total height from ground to tip of a blade at its apex in meters (m). |
| offshore      | number (integer) | Indicator of whether the turbine is offshore. 0—indicates turbine is not offshore. 1—indicates turbine is offshore. |
| retrofit      | number (integer) | Indicator of whether the turbine has been partially retrofit after initial construction (e.g., rotor and/or nacelle replacement). 0 indicates no known retrofit. 1 indicates yes known retrofit. |
| retrofit_year | number (integer) | Year in which the turbine was partially retrofit. |
| t_conf_atr    | number (integer) | Level of confidence in the turbine attributes. 1—No confidence: no attribute data beyond total height and year, 2—Partial confidence: incomplete information or substantial conflict between, 3—Full confidence: complete information, consistent across multiple data sources. |
| t_conf_loc    | number (integer) | Level of confidence in turbine location. 1—No turbine shown in image; image has clouds; imagery older than turbine built date, 2—Partial confidence: image shows a developed pad with concrete base and/or turbine parts on the ground, 3—Full confidence: image shows an installed turbine. |
| t_img_date    | number (integer) | Date of image used to visually verify turbine location. Note if source of image is NAIP, the month and day were set to 01/01. |
| t_img_srce    | string           | Source of image used to visually verify turbine location. |
| xlong         | number (float)   | Longitude of the turbine point, in decimal degrees. |
| ylat          | number (float)   | Latitude of the turbine point, in decimal degrees. |
| eia_id        | number (integer) | Plant ID from Energy Information Administration (EIA). |




