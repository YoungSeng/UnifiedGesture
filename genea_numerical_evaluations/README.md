# GENEA Numerical Evaluations
Scripts for numerical evaluations for the GENEA Gesture Generation Challenge:

https://genea-workshop.github.io/2020/#gesture-generation-challenge

This directory provides the scripts for quantitative evaluation of our gesture generation framework. We currently support the following measures:
- Average Jerk and Acceleration (AJ)
- Histogram of Moving Distance (HMD) for velocity and acceleration
- Hellinger distance between histograms
- Canonical Correlation Analysis (CCA) coefficient 
- Fréchet Gesture Distance (FGD)


## Obtain the data

Download the 3D coordinates of the GENEA Challenge systems at https://zenodo.org/record/4088319 .
Create a `data` folder and put challenge system motions there as in `data/Cond_X`.

## Run

`calk_jerk_or_acceleration.py`, `calc_histogram.py`, `hellinger_distance.py` and `calc_cca.py` support different quantitative measures, described below.


### Average jerk and acceleration

Average Jerk (AJ) represent the characteristics of gesture motion.

To calculate AJ, you can use `calk_jerk_or_acceleration.py`.

```sh
# Compute AJ
python calk_jerk_or_acceleration.py -m jerk -g your_prediction_dir
```

Note: `calk_jerk_or_acceleration.py` computes AJ for both original and predicted gestures. The AJ of the original gestures will be stored in `result/original` by default. The AJ of the predicted gestures will be stored in `result/your_prediction_dir`.

The same script can be used to calculate average acceleration (AA):

```sh
# Compute AA
python calk_jerk_or_acceleration.py -m acceleration -g your_prediction_dir
```


### Histogram of Moving Distance

Histogram of Moving Distance (HMD) shows the velocity/acceleration distribution of gesture motion.

To calculate HMD, you can use `calc_histogram.py`.
You can select the measure to compute by `--measure` or `-m` option (default: velocity).  
In addition, this script supports histogram visualization. To enable visualization, use `--visualize` or `-v` option.

```sh
# Compute velocity histogram
python calc_distance.py -g your_prediction_dir -m velocity -w 0.05  # You can change the bin width of the histogram

# Compute acceleration histogram
python calc_distance.py -g your_prediction_dir -m acceleration -w 0.05
```

Note: `calc_distance.py` computes HMD for both original and predicted gestures. The HMD of the original gestures will be stored in `result/original` by default.

### Hellingere distance

Hellinger distance indicates how close two histograms are to each other.

To calculate Hellinger distance, you can use `hellinger_distance.py` script.

### Canonical Correlation Analysis

Canonical Correlation Analysis (CCA) is a way of inferring information from cross-covariance matrices. If we have two vectors X = (X1, ..., Xn) and Y = (Y1, ..., Ym) of random variables, and there are correlations among the variables, then canonical-correlation analysis will find linear combinations of X and Y which have maximum correlation with each other.

To calculate CCA coefficient, you can use `calc_cca.py` script.

### Fréchet Gesture Distance
Please see [README](../../baselines/Tri/scripts/FGD/README.md) in the FGD folder.
