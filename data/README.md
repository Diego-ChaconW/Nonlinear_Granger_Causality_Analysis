# Data Directory

This project uses **synthetically generated data** from four chaotic maps
(Hénon, Ikeda, Tinkerbell, Rulkov). No external datasets are required.

Time series are generated programmatically in the experiment pipeline using
the functions in `src/maps.py`. Each run generates fresh data from the
chaotic map with fixed initial conditions, ensuring reproducibility.
