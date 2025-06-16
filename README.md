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
