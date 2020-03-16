# Automated Orthogonal Grid Generation for Regional Ocean Modeling with Schwarz-Chrisotffel Conformal Mappings

## Setup
### Dependencies
* [pyshp](https://pypi.org/project/pyshp/)
* [pygridgen](https://github.com/hetland/pygridgen)
* [lightgbm](https://pypi.org/project/lightgbm/)(best choice) or [Tensorflow](https://www.tensorflow.org/) or [Keras](https://pypi.org/project/Keras/) 
* [basemap](https://basemaptutorial.readthedocs.io/en/latest/)
* netCDF4


## Usage
You need to specify the path of the file which have the boundary coordinate of target(csv) in `./model` and the shape of the grid(like 50x50). After that, run the command:
```
python main.py --boundary_name [FILENAME] --shape1 [NUM1] --shape2 [NUM2]
```
*Example:*
```
python main.py --boundary_name boundary_bh_0.5d.csv --shape1 50 --shape2 50
```
The image and `.nc` file can be found in `./result/res`.

## Toy Example

Here is an illustration of the Bohai Sea(China): 
![image](https://github.com/gongbell/Automated-Orthogonal-Grid-Generation/blob/master/result/bh.png)
you also can download other ETOPO1(bedrock) data from [Grid Extract](https://maps.ngdc.noaa.gov/viewers/wcs-client/). For example, you can choose Bohai Sea(China) region and download its NetCDF file. Put it into `./shp` and then run python `get_boundary_bh.py` to obtain a new `.csv` file for `main.py`. 

## Train a new Model
if you want to make a new classification model, you can download the geographic boundary data from [GADM](https://www.gadm.org/) which are the natural and enough simple-polygon data. Put `.shp` and `.dbf` files we needed in `./shp` and run the command:
```
python train_data_process.py --node_num [NUMBER OF VERTEX]
```
*Example:*
```
python train_data_process.py --node_num 10
```
after that we can get 10-polygon data in `./shp/data_rough`. run the following command and we can get the train data for the base classificaiton model in `./shp/train_data`(the number of vertex is 10):
```
python train_data_get.py
```
In `./model` we have finished the base model named `lgb_model_10corenr_9adj.pkl` by `train_lightgbm.py`(need to modify some program statements). In order to make a new training data for powerful classificaiton model, e.g., we want to make a new model for 15-polygon with 10-polygon data. First, run `python train_data_process.py --node_num 15`  and then run: 
```
python make_new_classification.py --modelname lgb_model_10corenr_9adj.pkl --adj_num 9 --From 10 --To 15
```
actually,the effect of this command is similar to `train_data_get.py`. Last but not least, we can obtain a new calssification model through the command below.
```
python train_lightgbm.py --adj_num 15 --From 10 --To 15
```
