////////1. Define the area of Interest/////
var aoi = ee.FeatureCollection('projects/ee-agilakbar/assets/Border_GRT')
///////2. Prepared the Landsat Covariate data////
///2.1 Scale factor function for Landsat 8
function applyScaleFactors(image) {
  var opticalBands = image.select(['SR_B.*']).multiply(0.0000275).add(-0.2);
  var thermalBands = image.select(['ST_B.*']).multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true).addBands(thermalBands, null, true);
}

/// 2.2 Cloud Masking Function
function maskLandsatC2SR(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBitMask = 1 << 3;
  var shadowBitMask = 1 << 4;
  var snowBitMask = 1 << 5;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
              .and(qa.bitwiseAnd(shadowBitMask).eq(0))
              .and(qa.bitwiseAnd(snowBitMask).eq(0));
  return image.updateMask(mask);
}

///2.3 Defining the Landsat Data
var l8dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterDate('2018-01-01', '2018-12-30')
              .filterMetadata('CLOUD_COVER', 'less_than', 30)
//2.3.1 Applying the scale factors and cloud mask
var l8scaled = l8dataset.map(applyScaleFactors) // Applying the scale factors
var l8scaledmasked = l8scaled.map(maskLandsatC2SR) // Applying the Cloud Masking Function
                    .median() // Create the Median Compoisite
                    .clip(aoi) // Clip to the area of interest
                    .select(['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'],
                    ['B1','B2','B3','B4','B5','B6','B7']); // Renaming the bands
//Visualized the data
var visualization = {
  bands: ['B4', 'B3', 'B2'],
  min: 0.0,
  max: 0.3,
};
Map.addLayer(l8scaledmasked, visualization, 'True Color (432)')
Map.centerObject(aoi)
///2.4 Spectral Indicies Calculation
var spectral_indices_calculation = function(image) {
//Calculate the Green Normalized Difference Vegetation Index 
  var NDVI = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
//Calculate the Soil Adjusted Vegetation Index
  var savi = image.expression('(1 + L) * float(nir - red)/(nir + red + L)', {
            'nir': image.select('B5'),
            'red': image.select('B4'),
            'L': 0.9
            }).rename('SAVI')
//Calculate the Normalized Burn Index
  var nbr = image.normalizedDifference(['B5', 'B7']).rename('NBR')
//Calculate the Enhance Vegetation Index 2  
  var EVI2 = image.expression(
    '(2.5 * float(nir - red)/(nir + 2.4 * red + 1))', {
      'nir': image.select('B5'),
      'red': image.select('B4'),
    }).rename('EVI2');
//Calculate  Normalized Difference Water Index  
  var NDWI = image.normalizedDifference(['B5', 'B6']).rename('NDWI')
// Stack the spectral indexes
  var stack = NDVI.addBands(savi).addBands(nbr).addBands(EVI2).addBands(NDWI).toFloat()
  return stack;
};

//2.4.1 Applying the Spectral Indices Function
var spectral_ind = spectral_indices_calculation(l8scaledmasked)
print(spectral_ind, 'Landsat 8 Spectral Indices')

///////3. Prepared the Terrain, Sentinel-1////
///3.1 Terrain Attribute
var DEM = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename('DEM')
var slope = ee.Terrain.slope(DEM).rename('Slope') //Calculate the Slope
var aspect = ee.Terrain.aspect(DEM).rename('Aspect') // Calculate the ASpect

// 3.2 Define Sentinel-1 collection (VV and VH)
var S1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2018-01-01', '2018-12-31')
  .filterMetadata('instrumentMode', 'equals', 'IW')
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);
/// 3.2.1 Resampled to Match the spatial resolution of Landsat data
//  median compositing to reduce speckle noise
var S1_median = S1_collection.median();
// Resample to 30 m 
var S1_resampled = S1_median
  .resample('bilinear')
  .reproject({
    crs: 'EPSG:4326', 
    scale: 30
  });
// Clip
var S1_final = S1_resampled.clip(aoi);

///////4. Define the ancilarry data//
var distToRoad = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_road_rbi_main_country").rename('dist_road').clip(aoi);             
var distToRiver = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_river_rbi_country").rename('dist_river').clip(aoi);
var distToSettlement = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_settlement_rbi_country").rename('dist_settlement').clip(aoi);

//////5. Stacking the imagery
var covariates_2018 = ee.Image.cat(l8scaledmasked, 
                      spectral_ind, 
                      DEM, slope, aspect, 
                      S1_final,
                      distToRoad,
                      distToSettlement,
                      distToRiver)
print(covariates_2018)
///////4. Extract the Training data////
//Defining the Land Mask used by the authors
var land_mask = ee.Image("users/hadicu06/IIASA/RESTORE/miscellaneous/land_mask"); 
//4.1 Creating the function for extracting the training data based on crowdsource sample points
function extract_covariates_at_points(covariates_image, sample_points) {
  var res = sample_points
            .map(function(feature) {
              return ee.Feature(feature.geometry(),
                  covariates_image.reduceRegion({
                    reducer: ee.Reducer.mean(),
                    geometry: feature.geometry(),
                    scale: land_mask.projection().nominalScale(),
                    crs: land_mask.projection(),
                    maxPixels: 1e13,
                    tileScale: 4
                  }))
                  .copyProperties(feature);
            })
  
  res = res.filter(ee.Filter.neq('blue_median', null))
  return res;
}
///4.2 Defining the Sample points for each class
var samples_simplified_c1 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/01_undisturbedForest");
var samples_simplified_c2 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/02_loggedOverForest");  
var samples_simplified_c3 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/03_oilPalmMonoculture"); 
var samples_simplified_c4 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/04_treeBasedNotOilPalm");  
var samples_simplified_c5 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/05_cropland");  
var samples_simplified_c6 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/06_shrub"); 
var samples_simplified_c7 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/07_grassAndSavanna");  
var samples_simplified_c8 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/08_waterbody");  
var samples_simplified_c9 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/09_settlement");  
var samples_simplified_c10 = ee.FeatureCollection("users/hadicu06/IIASA/RESTORE/training_samples/simplified_class_crowdsourced/country/10_clearedLand");
///4.3 Applying the pixel value extraction function for sample points for each class
var features_simplified_c1 = extract_covariates_at_points(covariates_2018, samples_simplified_c1);
var features_simplified_c2 = extract_covariates_at_points(covariates_2018, samples_simplified_c2);
var features_simplified_c3 = extract_covariates_at_points(covariates_2018, samples_simplified_c3);
var features_simplified_c4 = extract_covariates_at_points(covariates_2018, samples_simplified_c4);
var features_simplified_c5 = extract_covariates_at_points(covariates_2018, samples_simplified_c5);
var features_simplified_c6 = extract_covariates_at_points(covariates_2018, samples_simplified_c6);
var features_simplified_c7 = extract_covariates_at_points(covariates_2018, samples_simplified_c7);
var features_simplified_c8 = extract_covariates_at_points(covariates_2018, samples_simplified_c8);
var features_simplified_c9 = extract_covariates_at_points(covariates_2018, samples_simplified_c9);
var features_simplified_c10 = extract_covariates_at_points(covariates_2018, samples_simplified_c10);


print("features_simplified_c1.limit(3)", features_simplified_c1.limit(3))


/////5. Implementing the classification

var params = {
  includeS1Features: false,  // NOTE: As including S1 features was not found to benefit overall in balance for all classes, and due to time difference between when dense S1 data started to be available and the reference map, all results were based on not including S1 features. 
  numTrees: 100,  
  minLeaf: 5,             
  seed: 42,                                    
  outScale: land_mask.projection().nominalScale().getInfo(),                  
  outProjection: land_mask.projection(),                              
  outCrs: land_mask.projection().crs(),
};
function train_classify_prob(training_samples_features, class_property_name, covariates_list, image_to_classify) {
  
  
  // Instantiate classifier
  var probClassifier = ee.Classifier.smileRandomForest({   
                          numberOfTrees: params.numTrees, 
                          minLeafPopulation: params.minLeaf,                
                          seed: params.seed,                                                   
                          variablesPerSplit: ee.List(covariates_list).size().divide(3).round().int().getInfo()  // ouch
                        }).setOutputMode('PROBABILITY');          


   // Train the classifier for classId primitive
   var trainedClassifier = probClassifier.train({
     features: training_samples_features, 
     classProperty: class_property_name, 
     inputProperties: covariates_list
   })
   
   // Apply the trained classifier to all pixels
   var classified = image_to_classify
                    .classify(trainedClassifier)
                    
   // Scale the value to integer (percent probability) for efficient storage
   classified = classified.multiply(100).round().byte()  
                
   return ee.Image(classified).rename('percent_probability');
  
}

function make_class_property_num(ft){
  return ft.set('class_binary_num', ee.Number.parse(ft.getString('class_binary_str')))
}

/// 5.1 Applied the Clasisification function
var covariates_2018_garut = covariates_2018.clip(aoi)

// 5.1.1 running classification on class 1

var training_samples = samples_class1.map(make_class_property_num);

var prob_class1 = train_classify_prob(
  training_samples,
  'class_binary_num',
  covariates_2018_garut.bandNames(),
  covariates_2018_garut
);
