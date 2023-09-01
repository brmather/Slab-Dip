### Converted from `_README.txt`

#### For a list of potential issues with integrating this model into the main EarthByte model, see this document: [Clennett_etal_2020](https://docs.google.com/document/d/1DbSrTRR3TFW-2Xd4QONjTbsEPZKXi7H4UOFSrpkIa4U/edit?usp=sharing)

These files comprise our model implemented into the Müller et al. (2019) reference frame. The types of files found here include: coastlines, plate boundaries, plate topologies, a rotation file and terrane shapefiles.

To view the models, open GPlates (downloadable at: [www.gplates.org](https://www.gplates.org)), click 'File' > 'Open Project', navigate to the folder containing the desired model, and then click on the file `Clennett_etal_2020.gproj`. This will simultaneously open all the files that comprise the model. A layers panel will appear, with the option to turn on/off certain files. The view can be changed by clicking on the globe, and the model can be run by clicking the play button in the animation bar, starting from 170Ma. Features can be inspected by clicking the 'choose feature' tab, selecting a feature, and clicking 'query feature'.

The files that comprise the model are described below:

* `Clennett_etal_2020_Coastlines.gpml`: Coastlines used in the reconstruction. The coastlines of western North America and Mexico were edited from the global model to account for later terrane accretions.
* `Clennett_etal_2020_Nam_boundaries.gpml`: File containing the new plate boundaries digitised in this study.
* `Clennett_etal_2020_Plates.gpml`: File containing the edited plate boundaries of the global model, as well as our new continuously-closing plate topologies.
* `Clennett_etal_2020_Rotations.rot`: This is the rotation file that contains the relative motions between plates, terranes and plate boundaries for western North America and the eastern Pacific basin. The first column specifies the plate ID, the second column the timestep, the third, fourth and fifth columns are the latitude, longitude and angle of the stage rotations, and the sixth column is the plate that the feature moves relative to. Most lines are accompanied with a comment describing the rotation.
* `Clennett_etal_2020_Terranes.gpml`: This file contains all the terranes shown in the model. We further divided these into superterranes, so that each can be coloured accordingly for better visualisation purposes:
    * `Angayucham.gpml`
    * `Farallon.gpml`
    * `Guerrero.gpml`
    * `Insular.gpml`
    * `Intermontane.gpml`
    * `Kula.gpml`
    * `North_America.gpml`
    * `Western_Jurassic.gpml`

`The Global_250-0Ma_Rotations_2019_v2.rot` and the files in the `DeformingMeshes` folder are unchanged from Müller et al. (2019).

For more information, read the GPlates manual which can be downloaded from [www.gplates.org](https://www.gplates.org) or [http://www.earthbyte.org/Resources/earthbyte_gplates.html](http://www.earthbyte.org/Resources/earthbyte_gplates.html)
