# Predict slab dip

Predict the dip angle of subducting oceanic lithosphere using simple plate kinematic parameters.

#### Cite

> Mather _et al_. (2022) "Kimberlite eruptions driven by slab flux and subduction angle". _Scientific Reports_. (in review)


## Dependencies

To run the Jupyter notebooks some dependencies are required:

- [pygplates](https://www.gplates.org/download/)
- [gplately](https://github.com/GPlates/gplately)
- [PlateTectonicTools](https://github.com/EarthByte/PlateTectonicTools/tree/master/ptt)
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) (for mapping)
- [netCDF4](https://pypi.org/project/netCDF4/) (to extract age grids of the seafloor)

Instructions to install these dependencies can be found within each package above.
Some conda instructions for setting up a Python environment are [here](https://www.benmather.info/post/2022-07-07-python-for-mac-m1/). While these have been written with the Mac M1 architecture in mind, the same instructions should apply equally to other distributions.

## Installation

Most of the Jupyter notebooks can be run without installing this package, however, following these installation instructions will make the slab dip prediction tool available system-wide.

From the current directory, run

```sh
pip install .
```

You can also install the most up-to-date version by running

```sh
pip install git+https://github.com/brmather/Slab-Dip.git
```

which will clone the `main` branch and install the latest version.

## Data packages

Plate reconstruction and corresponding age grids of the seafloor are required to predict slab dip. These may be downloaded from https://www.earthbyte.org/gplates-2-3-software-and-data-sets/

The slab dip prediction tool has been tested on [Clennett _et al._ (2020)](https://doi.org/10.1029/2020GC009117) and [Müller _et al._ (2019)](https://doi.org/10.1029/2018TC005462) plate reconstructions but should also work fine for all other plate reconstructions.


#### References

- Grose, C. J. (2012). Properties of oceanic lithosphere: Revised plate cooling model predictions. Earth and Planetary Science Letters, 333–334, 250–264. https://doi.org/10.1016/j.epsl.2012.03.037
- Hayes, G. P., Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., & Smoczyk, G. M. (2018). Slab2, a comprehensive subduction zone geometry model. Science, 362(6410), 58–61. https://doi.org/10.1126/science.aat4723
- Clennett, E. J., Sigloch, K., Mihalynuk, M. G., Seton, M., Henderson, M. A., Hosseini, K., et al. (2020). A Quantitative Tomotectonic Plate Reconstruction of Western North America and the Eastern Pacific Basin. Geochemistry, Geophysics, Geosystems, 21(8), 1–25. https://doi.org/10.1029/2020GC009117
- Müller, R. D., Zahirovic, S., Williams, S. E., Cannon, J., Seton, M., Bower, D. J., et al. (2019). A Global Plate Model Including Lithospheric Deformation Along Major Rifts and Orogens Since the Triassic. Tectonics, 38(6), 1884–1907. https://doi.org/10.1029/2018TC005462
