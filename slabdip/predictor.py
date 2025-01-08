import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import numpy as np
import gplately
import pkg_resources

DEFAULT_TESSELLATION = np.deg2rad(0.5)

subduction_data_filename = pkg_resources.resource_filename("slabdip", "data/subduction_data.csv")

default_DataFrame = pd.read_csv(subduction_data_filename, index_col=0)
default_DataFrame = default_DataFrame[np.isfinite(default_DataFrame['slab_dip'])]
default_variables = [
    'angle',
    'total_vel',
    'vel',
    'trench_vel',
    'vratio',
    'slab_age',
    'slab_thickness',
    'spreading_rate',
    'density',
]

def smooth_1D(array, sigma):
    if sigma > 0:
        return gaussian_filter1d(array, sigma)
    else:
        return array

def weighted_av(values, weights):
    """
    Take the weighted average of values 
    
    Some error checking to deal with empty arrays
    """
    if values.any():
        if len(values) > 1:
            return (values*weights).sum()/weights.sum()
        else:
            return values

class SlabDipper(object):
    
    def __init__(self, sklearn_regressor=None, X=None, y=None):
        
        self.scaler = StandardScaler()

        # set up regressor
        if sklearn_regressor is None:
            sklearn_regressor = KNeighborsRegressor(n_neighbors=3)

        self.kernel = sklearn_regressor

        # get training data from package directory
        if X is None:
            X = default_DataFrame[default_variables]
        if y is None:
            y = default_DataFrame['slab_dip']

        # scale the training data and train the neural network
        self.add_training_data(X, y)

        # initialise placeholder for gplately plate reconstruction model
        self._model = None
        self.downloader = None
        self.agegrid_filename = None
        self.spreadrate_filename = None
        
    def add_training_data(self, X, y):
        assert X.shape[0] == y.size, "X must have the same number of rows as y"

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.X = X
        self.X_scaled = X_scaled
        self.y = y

        self.kernel.fit(X_scaled, y)

        self.predictive_variables = X.columns
        self.tree = cKDTree(self.X)

    def get_score(self):
        return self.kernel.score(self.X_scaled, self.y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_predict = self.kernel.predict(X_scaled)
        return y_predict

    def add_plate_reconstruction(self, model):
        self._model = model

        if model.name:
            # initialise DataServer object
            self.downloader = gplately.DataServer(str(model.name))

    def set_age_grid_filename(self, filename):
        self.agegrid_filename = str(filename)

    def set_spreading_rate_grid_filename(self, filename):
        self.spreadrate_filename = str(filename)

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self.add_plate_reconstruction(value)
        
    def sample_age_grid(
            self,
            lons, 
            lats, 
            time, 
            from_rotation_reference_plate=None,
            to_rotation_reference_plate=None
        ):
        # age_grid = self.downloader.get_age_grid(reconstruction_time)
        if self.agegrid_filename:
            age_raster = gplately.Raster(data=self.agegrid_filename.format(time))
        elif self.downloader:
            grid = self.downloader.get_age_grid(time)
            age_raster = gplately.Raster(data=grid, extent=[-180,180,-90,90])
        else:
            raise ValueError("Cannot download age grid. \
                Set agegrid_filename or provide a supported reconstruction model")

        age_raster.fill_NaNs(inplace=True) # fill in NaN values

        # rotate grids if needed
        if from_rotation_reference_plate is not None and to_rotation_reference_plate is not None:
            spacingX = np.diff(age_raster.lons).mean()
            spacingY = np.diff(age_raster.lats).mean()
            mean_spacing = 0.5*(spacingX + spacingY)
            age_raster = age_raster.rotate_reference_frames(
                mean_spacing,
                time,
                from_rotation_features_or_model=self._model.rotation_model,
                to_rotation_features_or_model=self._model.rotation_model,
                from_rotation_reference_plate=from_rotation_reference_plate,
                to_rotation_reference_plate=to_rotation_reference_plate,
            )

        age_interp = age_raster.interpolate(lons, lats) # interpolate to trenches
        return age_interp
    
    def sample_spreading_rate_grid(
            self,
            lons, 
            lats, 
            time, 
            from_rotation_reference_plate=None,
            to_rotation_reference_plate=None
        ):
        # spreadrate_grid = self.downloader.get_spreading_rate_grid(reconstruction_time)
        if self.spreadrate_filename:
            spreadrate_raster = gplately.Raster(data=self.spreadrate_filename.format(time))
        elif self.downloader:
            grid = self.downloader.get_spreading_rate_grid(time)
            spreadrate_raster = gplately.Raster(data=grid, extent=[-180,180,-90,90])
        else:
            raise ValueError("Cannot download spreading rate grid. \
                Set spreadrate_filename or provide a supported reconstruction model")

        spreadrate_raster.fill_NaNs(inplace=True)

        # rotate grids if needed
        if from_rotation_reference_plate is not None and to_rotation_reference_plate is not None:
            spacingX = np.diff(spreadrate_raster.lons).mean()
            spacingY = np.diff(spreadrate_raster.lats).mean()
            mean_spacing = 0.5*(spacingX + spacingY)
            spreadrate_raster = spreadrate_raster.rotate_reference_frames(
                mean_spacing,
                time,
                from_rotation_features_or_model=self._model.rotation_model,
                to_rotation_features_or_model=self._model.rotation_model,
                from_rotation_reference_plate=from_rotation_reference_plate,
                to_rotation_reference_plate=to_rotation_reference_plate,
            )

        spreadrate_interp = spreadrate_raster.interpolate(lons, lats)
        return spreadrate_interp*1e-3
    
    def calculate_plate_density(self, plate_thickness, return_relative_density=False):
        h_slab = plate_thickness
        h_c = 7e3 # thickness of crust
        h_s = 43e3 # thickness of spinel field
        h_g = 55e3 # thickness of garnet field
        h_total = h_c + h_s + h_g

        rho_a = 3300
        rho_c = 2900 # density of crust
        rho_s = 3330 # density of spinel
        rho_g0, rho_g1 = 3370, 3340 # density of garnet (upper, lower)

        h_c = np.minimum(h_slab, h_c)
        h_s = np.minimum(h_slab - h_c, h_s)
        h_g = h_slab - h_c - h_s

        # find the density of the garnet field
        # a linear decay from g0 to g1 from 0 to 55+ km thickness
        def rho_garnet(h):
            m = -(3370 - 3340)/55e3
            c = 3370
            return h*m + c

        rho_g = 0.5*(rho_g0 + rho_garnet(h_g))

        rho_plate = (rho_c*h_c + rho_s*h_s + rho_g*h_g)/(h_c + h_s + h_g + 1e-6)
        rho_plate[np.isclose(h_c + h_s + h_g, 0)] = rho_c
        delta_rho = rho_plate - rho_a
        
        if return_relative_density:
            return rho_plate, delta_rho
        else:
            return rho_plate

        
    def segmentise_trench_boundaries(self, subduction_lon, subduction_lat):

        earth_radius = gplately.EARTH_RADIUS

        dtol = self.tessellation_threshold_radians*earth_radius + 5.0 # km
        segment_IDs = np.zeros(len(subduction_lon), dtype=int)

        index = 1
        for i in range(0, len(subduction_lon)-1):
            lon0 = subduction_lon[i]
            lat0 = subduction_lat[i]
            lon1 = subduction_lon[i+1]
            lat1 = subduction_lat[i+1]

            # distance between points (convert to unit sphere)

            xs, ys, zs = gplately.tools.lonlat2xyz([lon0,lon1], [lat0,lat1])
            dist = np.sqrt((xs[1]-xs[0])**2 + (ys[1]-ys[0])**2 + (zs[1]-zs[0])**2) * earth_radius

            if dist < dtol:
                # add to current segment
                segment_IDs[i] = index

            elif np.count_nonzero(segment_IDs == index) > 1:
                segment_IDs[i] = index
                index += 1
            else:
                pass

        unique_segment_IDs = set(segment_IDs)
        unique_segment_IDs.remove(0)
        return segment_IDs, unique_segment_IDs
    
    def calculate_trench_curvature(self, lons, lats, norm, length, smoothing=5, return_segment_IDs=False):
        
        subduction_norm = norm
        segment_IDs, unique_segment_IDs = self.segmentise_trench_boundaries(lons, lats)
    
        subduction_radius = np.zeros(len(lons))

        for i, seg_ID in enumerate(unique_segment_IDs):
            mask_segment = segment_IDs == seg_ID

            if np.count_nonzero(mask_segment) > 1:
                segment_norm = subduction_norm[mask_segment]

                # calculate angle between subuction zone segments
                dangle = np.gradient((segment_norm))

                # correct changes in plane
                dangle[dangle < 180] += 360
                dangle[dangle > 180] -= 360
                dangle[dangle < 90]  += 180
                dangle[dangle > 90]  -= 180

                distance = length[mask_segment]
                radius = dangle / distance

                # apply some smoothing
                smooth_radius = smooth_1D(radius, smoothing)
                subduction_radius[mask_segment] = smooth_radius
            else:
                subduction_radius[mask_segment] = 0
            
        
        if return_segment_IDs:
            return subduction_radius, segment_IDs, unique_segment_IDs
        else:
            return subduction_radius
    
    def tessellate_slab_dip(
            self,
            time,
            tessellation_threshold_radians=DEFAULT_TESSELLATION,
            from_rotation_reference_plate=None,
            to_rotation_reference_plate=None
        ):
        time = int(time)
        self.tessellation_threshold_radians = tessellation_threshold_radians

        if self._model is None:
            raise ValueError("Don't forget to set a GPlately plate model! `self.model = model`")

        # update plate ID
        if to_rotation_reference_plate is not None:
            self._model.anchor_plate_id = to_rotation_reference_plate

        subduction_data = self._model.tessellate_subduction_zones(
                                                           time,
                                                           tessellation_threshold_radians,
                                                           ignore_warnings=True,
                                                           output_subducting_absolute_velocity_components=True)

        subduction_lon         = subduction_data[:,0]
        subduction_lat         = subduction_data[:,1]
        subduction_vel         = subduction_data[:,2]*1e-2
        subduction_angle       = subduction_data[:,3]
        subduction_norm        = subduction_data[:,7]
        subduction_pid_sub     = subduction_data[:,8]
        subduction_pid_over    = subduction_data[:,9]
        subduction_length      = np.deg2rad(subduction_data[:,6]) * gplately.EARTH_RADIUS * 1e3 # in metres
        subduction_convergence = np.fabs(subduction_data[:,2])*1e-2 * np.cos(np.deg2rad(subduction_data[:,3]))
        subduction_migration   = np.fabs(subduction_data[:,4])*1e-2 * np.cos(np.deg2rad(subduction_data[:,5]))
        subduction_plate_vel   = subduction_data[:,10]*1e-2

        subduction_convergence = np.clip(subduction_convergence, 0, 1e99)

        # project points away from the trench (edge of plate boundary)
        subduction_lon_buffer = subduction_lon - np.sin(np.deg2rad(subduction_norm))
        subduction_lat_buffer = subduction_lat - np.cos(np.deg2rad(subduction_norm))

        # sample AgeGrid
        age_interp = self.sample_age_grid(
            subduction_lon_buffer,
            subduction_lat_buffer,
            time,
            from_rotation_reference_plate=from_rotation_reference_plate,
            to_rotation_reference_plate=to_rotation_reference_plate)

        # calculate the thickness of the downgoing plate
        thickness = gplately.tools.plate_isotherm_depth(age_interp)

        # sample spreadrate grid
        spreadrate_interp = self.sample_spreading_rate_grid(
            subduction_lon_buffer,
            subduction_lat_buffer,
            time,
            from_rotation_reference_plate=from_rotation_reference_plate,
            to_rotation_reference_plate=to_rotation_reference_plate)

        # get the ratio of convergence velocity to trench migration
        vratio = (subduction_convergence + subduction_migration)/(subduction_convergence + 1e-22)
        vratio[subduction_plate_vel < 0] *= -1
        vratio = np.clip(vratio, 0.0, 1.0)
        
        subduction_flux = subduction_convergence*thickness

        # calculate density of the down-going plate
        rho_plate, delta_rho = self.calculate_plate_density(thickness, return_relative_density=True)
        
        # calculate trench curvature
        subduction_radius, segment_IDs, unique_segment_IDs = self.calculate_trench_curvature(
            subduction_lon, subduction_lat, subduction_norm, subduction_length, return_segment_IDs=True)
        
        # stuff all these variables into a DataFrame
        output_data = np.column_stack([
            subduction_lon,
            subduction_lat,
            subduction_angle,
            subduction_norm,
            subduction_pid_sub,
            subduction_pid_over,
            subduction_length,
            subduction_vel,
            subduction_convergence,
            subduction_migration,
            subduction_plate_vel,
            subduction_flux,
            age_interp,
            thickness,
            vratio,
            segment_IDs,
            subduction_radius,
            rho_plate,
            delta_rho,
            spreadrate_interp
        ])

        header = ['lon', 'lat', 'angle', 'norm', 'pid_sub', 'pid_over', 'length', 
                  'total_vel', 'vel', 'trench_vel', 'slab_vel_abs',
                  'slab_flux', 'slab_age', 'slab_thickness', 'vratio',
                  'segment_ID', 'curvature', 'density', 'relative_density', 'spreading_rate']

        df = pd.DataFrame(output_data, columns=header)
        df.dropna(inplace=True)
        df_X = df[self.predictive_variables]

        # calculate slab dip - clip to realistic range
        slab_dip = np.clip(self.predict(df_X), 0, 90)

        # find the euclidean distance between training data and predicted result
        distance, index = self.tree.query(df_X, p=2)

        df = df.assign(slab_dip=slab_dip, prediction_distance=distance, nearest_neighbour=index)
        return df

    def smooth_along_segments(self, df, variable_name, smoothing=5):

        # extract segments
        segment_IDs = df['segment_ID'].to_numpy().astype(int)
        unique_segment_IDs = np.unique(segment_IDs)

        # extract variable
        array = df[variable_name]
        smooth_array = array.copy()

        # smooth array along segment
        for i, seg_ID in enumerate(unique_segment_IDs):
            mask_segment = segment_IDs == seg_ID
            if np.count_nonzero(mask_segment) > 1:
                smooth_array[mask_segment] = smooth_1D(array[mask_segment], smoothing)

        return smooth_array

    def surrogate_slab_dip(self, df):

        # extract segments
        segment_IDs = df['segment_ID'].to_numpy().astype(int)
        unique_segment_IDs = np.unique(segment_IDs)

        dip = df['slab_dip'].copy()

        # smooth array along segment
        for i, seg_ID in enumerate(unique_segment_IDs):
            mask_segment = segment_IDs == seg_ID
            if np.count_nonzero(mask_segment) > 1:
                dip[mask_segment] = weighted_av(df['slab_dip'][mask_segment],
                                                df['prediction_distance'][mask_segment])

        return dip

