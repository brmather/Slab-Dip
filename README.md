# Predict slab dip

Predict the dip angle of subducting oceanic lithosphere using simple plate kinematic parameters.

#### Cite

```bib
@article{Mather2023,
  title = {Kimberlite Eruptions Driven by Slab Flux and Subduction Angle},
  author = {Mather, Ben R and M{\"u}ller, R Dietmar and Alfonso, Christopher P. and Seton, Maria and Wright, Nicky M.},
  year = {2023},
  journal = {Scientific Reports},
  volume = {13},
  number = {9216},
  pages = {1--12},
  doi = {10.1038/s41598-023-36250-w},
}
```

> Mather, B. R., Müller, R. D., Alfonso, C. P., Seton, M., & Wright, N. M. (2023). Kimberlite eruptions driven by slab flux and subduction angle. Scientific Reports, 13(9216), 1–12. https://doi.org/10.1038/s41598-023-36250-w



## Dependencies

To run the Jupyter notebooks some dependencies are required:

- [pygplates](https://www.gplates.org/download/)
- [gplately](https://github.com/GPlates/gplately)
- [PlateTectonicTools](https://github.com/EarthByte/PlateTectonicTools/tree/master/ptt)
- [Scikit-Learn](https://scikit-learn.org)
- [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) (for mapping)
- [netCDF4](https://pypi.org/project/netCDF4/) (to extract age grids of the seafloor)

Instructions to install these dependencies can be found within each package above.
Some conda instructions for setting up a Python environment are [here](https://www.benmather.info/post/2022-07-07-python-for-mac-m1/). While these have been written with the Mac M1 architecture in mind, the same instructions should apply equally to other distributions.

## Installation

Most of the Jupyter notebooks can be run without installing this package, however, following these installation instructions will make the slab dip prediction tool available system-wide.

### 1. Using conda (recommended)

You can install the latest stable public release of `slabdip` and all of its dependencies using conda.
This is the preferred method to install `slabdip` which downloads binaries from the conda-forge channel.

```sh
conda install -c conda-forge slabdip
```

#### Creating a new conda environment

We recommend creating a new conda environment inside which to install `slabdip`. This avoids any potential conflicts in your base Python environment. In the example below we create a new environment called "`my-env`":

```sh
conda create -n my-env
conda activate my-env
conda install -c conda-forge slabdip
```

`my-env` needs to be activated whenever you use `GPlately`: i.e. `conda activate my-env`.

### 2. Using pip

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

## Usage

A series of Jupyter notebooks document the workflow to calculate plate kinematic and rheological information used to predict slab dip. Skip to __notebook 6__ to jump straight into the slab dip estimator. The Python snippet below outlines the usage of the `SlabDipper` object which can be used with little modification to estimate slab dip for a user-defined reconstruction time.

```python
# Call GPlately's DataServer object and download the plate model
gdownload = gplately.download.DataServer("Clennett2020")
rotation_model, topology_features, static_polygons = gdownload.get_plate_reconstruction_files()

# Use the PlateReconstruction object to create a plate motion model
model = gplately.PlateReconstruction(rotation_model, topology_features, static_polygons)

# Initialise SlabDipper object
dipper = SlabDipper()
dipper.model = model

# Set the filename (including path) of the seafloor age and spreading rate grids
dipper.set_age_grid_filename(agegrid_filename)
dipper.set_spreading_rate_grid_filename(spreadrate_filename)

# Estimate slab dip across the globe for a specified reconstruction time
# (returned as a Pandas DataFrame)
dataFrame = dipper.tessellate_slab_dip(0)
```

#### References

- Clennett, E. J., Sigloch, K., Mihalynuk, M. G., Seton, M., Henderson, M. A., Hosseini, K., et al. (2020). A Quantitative Tomotectonic Plate Reconstruction of Western North America and the Eastern Pacific Basin. Geochemistry, Geophysics, Geosystems, 21(8), 1–25. https://doi.org/10.1029/2020GC009117
- Müller, R. D., Zahirovic, S., Williams, S. E., Cannon, J., Seton, M., Bower, D. J., et al. (2019). A Global Plate Model Including Lithospheric Deformation Along Major Rifts and Orogens Since the Triassic. Tectonics, 38(6), 1884–1907. https://doi.org/10.1029/2018TC005462
