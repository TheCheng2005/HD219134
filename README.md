# HD219134
Modeling Work on HD219134 with EXOPLANET (Foreman-Mackey et al. 2021), paper coming soon...

Files in main are model building code. RV_LC_GP_4planet.py yields the best statistical results. We are still making some final tweaks to the model. These models produce a trace.nc file and a map.pkl file. These models also output BIC and log-likelihood model for comparison.

Files in Data are all the light curve and radial velocity data we collected from the literature. 

Files in Graph-Table-Making take in the trace.nc file and produce 1) corner, RV, LC, residual plots; 2) comparison to literature, best draw parameters spreadsheets

Files in LitData are the best fits results from the literature, taken from NASA's Exoplanet Archive
