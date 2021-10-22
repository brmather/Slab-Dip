# Predict slab dip

Predict the average slab dip of subducting oceanic lithosphere using the following relation,

![equation](https://user-images.githubusercontent.com/15915881/138374318-32650216-f131-44c3-b9db-229f28208d97.png)

where _h_ is the thickness of the subducting plate calculated by plate models of lithospheric cooling (from [Grose, 2012](https://doi.org/10.1016/j.epsl.2012.03.037)), _vc_ is the convergence velocity, _vt_ is the rate of trench migration, and _A_ and _B_ are coefficients obtained from a weighted linear regression.
Using the present-day configuration of subduction boundaries and slab dips obtained from the Slab2 model ([Hayes _et al._, 2018](https://doi.org/10.1126/science.aat4723)), we obtain _A_ = 0.006 and _B_ = 8 with standard deviations of _σA_ = 0.001 and _σB_ = 5.


## Dependencies

To run this notebook some dependencies are required:

- [pygplates](https://www.gplates.org/download/)
- [PlateTectonicTools](https://github.com/EarthByte/PlateTectonicTools/tree/master/ptt)
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) (for mapping)
- [netCDF4](https://pypi.org/project/netCDF4/) (to extract age grids of the seafloor)


## Data packages

A plate reconstruction and corresponding age grids of the seafloor are required to predict slab dip. These may be downloaded from https://www.earthbyte.org/gplates-2-3-software-and-data-sets/

The slab dip calculation has been tested on [Clennett _et al._ (2020)](https://doi.org/10.1029/2020GC009117) and [Müller _et al._ (2019)](https://doi.org/10.1029/2018TC005462) plate reconstructions but should also work fine for other plate reconstructions.


#### References

- Grose, C. J. (2012). Properties of oceanic lithosphere: Revised plate cooling model predictions. Earth and Planetary Science Letters, 333–334, 250–264. https://doi.org/10.1016/j.epsl.2012.03.037
- Hayes, G. P., Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M. (2018). Slab2, a comprehensive subduction zone geometry model. Science, 362(6410), 58–61. https://doi.org/10.1126/science.aat4723
- Clennett, E. J., Sigloch, K., Mihalynuk, M. G., Seton, M., Henderson, M. A., Hosseini, K., et al. (2020). A Quantitative Tomotectonic Plate Reconstruction of Western North America and the Eastern Pacific Basin. Geochemistry, Geophysics, Geosystems, 21(8), 1–25. https://doi.org/10.1029/2020GC009117
- Müller, R. D., Zahirovic, S., Williams, S. E., Cannon, J., Seton, M., Bower, D. J., et al. (2019). A Global Plate Model Including Lithospheric Deformation Along Major Rifts and Orogens Since the Triassic. Tectonics, 38(6), 1884–1907. https://doi.org/10.1029/2018TC005462
