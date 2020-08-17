# Off The Beaten Sidewalk: Pedestrian Prediction In Shared Spaces For Autonomous Vehicles
by Cyrus Anderson at [UM FCAV](https://fcav.engin.umich.edu/)

### Introduction
This paper presents the method Off the Sidewalk Predictions (OSP) to predict pedestrians'
trajectories in scenes where sidewalks and other traffic devices may not be
present (such as [shared spaces](https://en.wikipedia.org/wiki/Shared_space)).

arxiv: https://arxiv.org/abs/2006.00962

## Predict Trajectories

Predictions with pre-trained models can be made by running
```
python driver_low_mem.py
```

## Model Fitting
Model parameters can be estimated from data by running
```
python ss_model/fit_model_driver.py
```

## File Structure

The structure at `SAMPLE_DATASETS_ROOT`:
```
sample_data
   | tt_format
      | 10hz
          | dut
```
Additional datasets can be resampled and formatted with the tools in `utils/dataset_conversion.py`.
The pedestrian datasets used in the paper are from:
- [DUT](https://github.com/dongfang-steven-yang/vci-dataset-dut)
- [InD Dataset](https://www.ind-dataset.com) (apply there)


### Dependencies

- numpy
- scipy
- pandas
