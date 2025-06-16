# Crowdsource-Land-Cover-Mapping
## The Plan
Based on the works of [Hadi et al 2022](https://doi.org/10.1038/s41597-022-01689-5) which gather crowdsource based land cover reference data in Indonesia, I I designed my replication workflow based on three key stages:
1. Literature Study and Preparation
In this stage, I reviewed the scientific publication, technical documentation, and supplementary datasets (crowdsource data). I also examined the original Google Earth Engine (GEE) code repository structure. This stage allowed complete understanding the logical workflow and approach for conducting the classification, which consist of covariate/multisource remote sensing data, binary classification approach, and final generation of the land cover maps. In this stage, i manage to identify several possible challenges and rooms for improvement.  
2. Code Adaptation, replication, and modification
After preparing the necessary setup, such as google earth engine code editor and github repository, i start to modified the script based on my personal workflow,      and integrate them with the workflow of the crowdsource land cover mapping. I am planning to simplify the remote sensing data preparation, since the updated Landsat data in google earth engine allow several step in data preparation process to be simplfied. For example, since the original script used Landsat collection 1, which since deprecated, i will replace it with Landsat Collection 2 ARD data. As the result, several preprocessing step can be optimized for more effective workflow
3. Documentation
As requested, i will documented each steps i take in order to replicate the work, while identifying challenges that may arise from the workflow and possible sollution. Additionally, modification of the script will be documented as well as proposed improvement of the current classification workflow.
## The Code Adaptation
### Area of Interest and Remote Sensing Dataset Preparation
The first stage of modifying the code is defining the area of interest (AOI). Since most of my research is conducted in Garut Regency, West Java Province, I decided to conduct the classification for this area. The AOI definition is rather a simple approach, by using adaministrative boundary of Garut I already uploaded in Google Earth Engine. I define the AOI using the following script:
```Javascript
////////I. Define the area of Interest/////
var aoi = ee.FeatureCollection('projects/ee-agilakbar/assets/Border_GRT')
```
I am planning to used the Landsat Collection 2 Tier Surface Reflectance data (LANDSAT_LC08_C02_T1_L2), since it is an Analysis Ready Data (ARD), which requried minimal preprocessing. The main preprocessing is applying the scale factor for the data, and cloud masking procedure using the following script:
```Javascript
///////II. Prepared the Landsat multisource data////
/// Scale factor function for Landsat 9
function applyScaleFactors(image) {
  var opticalBands = image.select(['SR_B.*']).multiply(0.0000275).add(-0.2);
  var thermalBands = image.select(['ST_B.*']).multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true).addBands(thermalBands, null, true);
}

// Cloud Masking Function
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
```
### Creating the Composite and Covariate Landsat Data
Since the original scipt used a yearly composite, I also used the yearly composite of 2018, while using Landsat 8 data. I implement filter date and filter metadata command to acquired the less cloudy scene. The previously defined function (scale factors and cloud masked) is also applied, resulting in an image which has substantionally less cloud.
```Javascript
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
```
![image](https://github.com/user-attachments/assets/82a5f0fe-5af5-4b68-b4b7-29a94b14b636)
<br>

The original script used several spectral indices, which consist of NDVI, NBR, NDWI, SAVI, and EVI 2. To derived this indices, i create a function for automatically derived each indices and stacked into a seperate data.
```Javascript
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
```
### Deriving Terrain, Sentinel-1 data, and Ancillary Data
Based on the original scripts, the author also used multisource data, which consist of Terrain Attribute, Sentinel-1, and PALSAR textural features. Based on my experience and several literature studies, textural data only improved the accuracy of land cover map, if applied to high resolution imagery ([Chen et al 2004](https://doi.org/10.1080/01431160310001618464)). Additionally, the author argue that the classification scheme also plays critical role, and as the result, when applying the textural features on detailed classification scheme, did not improve the overall accuracy. Therefore, I decided not to implement the textural features for this replication, instead only used Terrain and Sentinel-1 data. Below are the implementation of this workflow:
```Javascript
///////3. Prepared the Terrain, Sentinel-1, and PALSAR data////
///3.1 Terrain Attribute
var DEM = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename('DEM')
var slope = ee.Terrain.slope(DEM).rename('Slope') //Calculate the Slope
var aspect = ee.Terrain.aspect(DEM).rename('Aspect') // Calculate the ASpect
```
Since Sentinel-1 have a spatial resolution of 10 m, which can cause a problem when stacked with 30m Landsat data. To mitigate the problem, I resampled the data to match the spatial resolution of Landsat data. Below are the implementation of the script:
```Javascript
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
```
In addition to remote sensing data, the authors also used other GIS data, namely distance to road, distance to river, and distance to settlement. Since i find that the data is accessible directly in earth engine, I just import these data into my workflow and clip them using my AOI
```Javascript
///////4. Define the ancilarry data//
var distToRoad = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_road_rbi_main_country").rename('dist_road').clip(aoi);             
var distToRiver = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_river_rbi_country").rename('dist_river').clip(aoi);
var distToSettlement = ee.Image("users/hadicu06/IIASA/RESTORE/covariates_images/distance_settlement_rbi_country").rename('dist_settlement').clip(aoi);
```
### Stacking all the predictor variables
After preparing these data, i used image concatenates (ee.image.cat) to stacked the predictor variables into a single image so i can extract the training data based on the crowdsource data and my AOI
```Javascript
//////5. Stacking the imagery
var covariates_2018 = ee.Image.cat(l8scaledmasked, 
                      spectral_ind, 
                      DEM, slope, aspect, 
                      S1_final,
                      distToRoad,
                      distToSettlement,
                      distToRiver)
```
## Extracting the Pixel Value for Training the Classifiers
Based on my analysis of the original code structure, the extraction of pixel values for training samples is conceptually similar to the standard workflow I typically apply for supervised classification. While the fundamental principle of extracting covariate values at sample points remains standard, this workflow incorporates careful handling of projection, scale, and data integrity to ensure compatibility across multisource datasets.
```Javascript
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
// Inspect the result
print("features_simplified_c1.limit(3)", features_simplified_c1.limit(3))
```
## Implementing the Supervised Classification
The supervised classification is conducted using the Random Forest classiifers. However, instead using hard classification output, the authors used the binary probability classification approach. Upon literature study, i found that this binary classification approach allow for expert-based post-processing, uncertainty management, and better control over misclassification. This approach is beneficial in my opnion since the crowdsource data may result in uncertainty which can be mitigated using the expert based post processing. For this example, i tried to implement the probability classification for one class only, with the following script:
```Javascript
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
```
At this stage, the classification function is fully implemented and ready to process the covariate stack and training data. Due to time limitation, full generation of binary classification and post processing of the binary probability is not conducted

## Challenges
During the replication of the workflow, I encounter several challenges and gap in the data preparation and classification.
1. Landsat Data Preparation
As mentioned before, the main data source (Landsat Covariates) is based on Landsat Collection 1 Level 1, which is depracted. As the result, complete replication of the preprocessing cannot implemented, especially, BRDF correction, since the Landsat Collection 2 already preprocessed. I have encounter this problem before during my research, but i found that Landsat data over land areas can be directly implemented, especially for land cover classification. Additionally, the original script conducted a terrain correction which i did not conducted. This decision is based on previous experience and supervision during my master's thesis, in which my supervisor stated that terrain correction is unnessary if the land cover phenomena did not affected by terrain position. Since i conducted the classification in Garut regency, i found that land cover phenomena in the region did not signficantly affected by terrain position.
2. Covariates/Predictor Variables
In the covariates data list, i believed that many important variables are not included, such as Thermal Bands, Tasseled Cap Transformation, and various other terrain attributes. Based on my experience, these predictor variables significantly improve the classification accuracy and visual representation of the classification result.
3. Google Earth Engine Limitation
During the trial and error for the script, I encounter errors in which the computation time of earth engine tends to time out during inspection of the feature collection. This problem reduce the capability of my troubleshooting since this errod could be accumulated in the final process of the classification. 
