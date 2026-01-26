# PREPBUFR Observation Types Summary

A summarized table grouping similar observation types together.

| Category | Codes | Message Type | Description | Approximate Active Period |
|----------|-------|--------------|-------------|---------------------------|
| **Weather Balloons** | 120, 220 | ADPUPA | Rawinsonde (weather balloon) - temperature, humidity, pressure, winds | 1975-present |
| | 221 | ADPUPA | Pilot balloon (PIBAL) - winds only | 1975-present |
| **Aircraft** | 130, 230 | AIRCFT | AIREP/PIREP manual pilot reports - temperature, winds | 1975-present |
| | 131, 231 | AIRCFT | AMDAR automated aircraft - temperature, winds | ~1985-present |
| | 133, 233 | AIRCAR | ACARS commercial aircraft - temperature, humidity, winds | ~1990-present |
| | 134, 234 | AIRCFT | TAMDAR regional aircraft - temperature, humidity, winds | ~2005-2015, 2020+ |
| | 135, 235 | AIRCFT | Canadian AMDAR aircraft - temperature, winds | ~2010-present |
| **Hurricane Recon** | 132, 232 | ADPUPA | Dropsonde & flight-level reconnaissance - temperature, humidity, winds | ~1980-present |
| | 182 | SFCSHP | Splash-level dropsonde (ocean surface) - temperature, humidity, pressure | 1975-present |
| **Surface Land** | 181, 281 | ADPSFC | SYNOP/METAR with station pressure - temperature, humidity, pressure, winds | 1975-present |
| | 183, 284 | ADPSFC, SFCSHP | Surface land/marine with missing station pressure | ~1980-present |
| | 187, 287 | ADPSFC | METAR with missing station pressure | ~1990-present |
| | 188 | MSONET | Surface mesonet observations | ~2000-2010 |
| **Surface Marine** | 180, 280 | SFCSHP | Ship/buoy/C-MAN with station pressure - temperature, humidity, pressure, winds | 1975-present |
| | 282 | SFCSHP | ATLAS buoy - winds | ~1985-present |
| | 191 | SFCBOG | Australian PAOB bogus sea-level pressure | ~1980-2010 |
| **Wind Profilers** | 223 | PROFLR | NOAA Profiler Network - winds | ~1990-2017 |
| | 224 | VADWND | NEXRAD VAD radar-derived winds | ~1995-present |
| | 228 | PROFLR | JMA wind profiler | ~2000-present |
| | 229 | PROFLR | European wind profiler (from PIBAL bulletins) | ~1995-present |
| | 126 | RASSDA | RASS temperature profiles from profilers | ~1995-2017 |
| **Satellite Winds (Geostationary)** | 245, 246, 251 | SATWND | GOES (US) - IR, water vapor, visible cloud drift winds | ~1995-present |
| | 242, 250, 252 | SATWND | JMA (Japan) - GMS/MTSAT/Himawari cloud drift winds | ~1995-present |
| | 243, 253, 254 | SATWND | EUMETSAT - Meteosat cloud drift & water vapor winds | ~1995-present |
| | 240, 241, 244, 255, 256 | SATWND | Other geostationary (GOES SW-IR, India, AVHRR) - various | ~1995-2015 |
| **Satellite Winds (Polar)** | 257, 258, 259 | SATWND | MODIS (Aqua/Terra) - IR & water vapor winds | ~2005-present |
| **Satellite Moisture** | 152 | SPSSMI | SSM/I precipitable water over ocean | ~1990-2010 |
| | 153 | GPSIPW | GPS-derived precipitable water | ~2000-present |
| **Scatterometer Winds** | 285 | QKSWND | QuikSCAT ocean surface winds | ~2000-2010 |
| | 286 | ERS1DA | ERS scatterometer ocean winds | ~1995-2005 |
| | 289 | WDSATR | WindSat ocean surface winds | ~2005-2015 |
| | 290 | ASCATW | ASCAT (METOP) ocean surface winds | ~2008-present |
| **Synthetic/Bogus** | 210 | SYNDAT | Synthetic tropical cyclone winds | ~1980-present |

## Sources

- NCEP PREPBUFR Report Types documentation (prepbufr_types.html, revised 2/13/2018)
- NNJA Conventional Sensors inventory (nnja_conventional.png, as of 01/25/2026)
