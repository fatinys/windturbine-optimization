# **Wind Turbine Capacity Optimzation**
<img src="reports\figures\app.png" alt="Description" width="1000" height="525">

## **CURRENT BUILD**
You can run the predictor locally by running `app.py`, deploying into production soon with Azure
--------------------------

## **Project Directory**

```
windturbine-optimization
├───data
│   ├───external
│   ├───interim
│   │   └───turbinelocsplit (API Dataframe Split)
│   ├───processed
│   └───raw
│       └───weatherlocsplit (API Dataframe Split)
├───notebooks
│   ├───turbineexploration (Exploration and Preliminary Modelling of Turbine Data)
│   └───windgeneration  (Exploration of Texas Wind Generation Data)
├───references
├───reports
│   └───figures
│       └───models (Figures of Model Evaluation)
├───src
│   ├───data (Data Acquisition and Preprocessing Scripts)
│   ├───modelling (Model Classes and Methods)
│   └───visualization (Data Visualization Scripts)
└───windturbineoptimization

```

## **Plan:**

We have two primary goals:
1. Create a model to predict the energy capacity for a turbine given its location.
2. Provide optimal zones in Texas that show potential for efficient wind farm projects.





# Project Overview

### **Data Collection**: 
The data we need isnt all in the same place so we'll need to source it from several places. In `src/data`, youll find the relevant code that was used to create the data.

1. **Turbine Data**: In order to cut down on variables, we'll limit the search for turbines made within the last 10 years, this will keep the data relevant and consistent. For the sake of simplicity we'll choose not to consider manufacturer and model. We are getting this data from The U.S. Wind Turbine Database: https://eerscmap.usgs.gov/uswtdb/.
2. **Historical Wind Data**: For each turbine that we collect, using their respective coordinates. This data is coming from, https://open-meteo.com/. This data comes in daily spans, and weather data means we can't just simply average features over a 365 day span. We will attempt to use some crafty statistical methods to attempt to correctly characterize the data in context.

### **Notebooks**
You will find notebooks going the data exploration and processing the data, as well as some preliminary modelling to establish a baseline. Find these in `notebooks`

### **Scripts**

Most of the exploratory work done for this project was done interactively through the VS code interactive window, however the codes and functions have been refactored and modularized for readability and reproduction. These you will find in `src`

### **Models**

The models used in this project are fitted to classes for ease of access. Each class has methods to evaluate the model, and to plot residual and vs graphs. Check comments in `src/modelling/modelling.py` for direction on how to use. 

# EDA Summary
### Wind generation
To pick the best months to pull weather data from, I wanted to understand when turbines are maximally efficient and produce the most energy. I was able to find wind energy generation data taken from the EIA(https://www.eia.gov/electricity/monthly/). I mapped this data and standardized it to account for technological advancements, to focus on the sole impact of wind and weather on energy generation.

<img src="reports\figures\windgeneration_monthly.png" alt="Description" width="1000" height="550">


We can clearly see a pattern in which energy generation is higher in months where there is more windflow and/or the wind is heavier. There is definitely a seasonal pattern to wind generation. From here we chose from around November through June to be the best time period to collect data.

### Data Preprocessing

<img src="reports\figures\turbinevar_corr.png" alt="Description" width="700" height="700">

We opt to remove `t_rsa` (Rotor Swept Area), because it is essentially a function of the rotor diameter. The rest of the variables are safe, doesn't seem to have much risk of multicollinearity.

We also removed outliers from the data via IQR.

### Before

<img src="reports\figures\before_processing.png" alt="Description" width="700" height="400">

### After

<img src="reports\figures\after_processing.png" alt="Description" width="700" height="400">








# Modelling Summary

After processing the data(cleaning outliers, getting rid of extra columns, combining data sources.) We split the data into training and test splits(70/30). 

After fine tuning, these were the best we could get the models to perform.

**Linear Regression:**

<img src="reports\figures\models\linreg_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\linreg_vs.png" alt="Description" width="300" height="200">

**Ridge Regression:**

<img src="reports\figures\models\ridge_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\ridge_vs.png" alt="Description" width="300" height="200">

**Lasso Regression:**

<img src="reports\figures\models\lasso_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\lasso_vs.png" alt="Description" width="300" height="200">

**Decision Tree Regression:**

<img src="reports\figures\models\decision_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\decision_vs.png" alt="Description" width="300" height="200">

**Random Forest Regression:**

<img src="reports\figures\models\forest_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\forest_vs.png" alt="Description" width="300" height="200">

**Gradient Boosting Regression:**

<img src="reports\figures\models\gradient_resid.png" alt="Description" width="300" height="200">
<img src="reports\figures\models\gradient_vs.png" alt="Description" width="300" height="200">

## **Performance Table**


|                   |       MAE |     RMSE |      r^2 |        MAPE |   Training Time |   Prediction Time |
|:------------------|----------:|---------:|---------:|------------:|----------------:|------------------:|
| Linear Regression | 241.714   | 318.542  | 0.858274 | 0.123826    |       0.146992  |         0.147993  |
| Ridge Regression  | 241.72    | 318.544  | 0.858273 | 0.123827    |       0.157085  |         0.158972  |
| Lasso Regression  | 241.714   | 318.542  | 0.858274 | 0.123826    |       0.159324  |         0.160324  |
| Decision Tree     |   1.33017 |  21.5777 | 0.99935  | 0.000783815 |       0.0316479 |         0.0326474 |
| Random Forest     |   1.51449 |  21.3611 | 0.999363 | 0.00109383  |       1.43843   |         1.45443   |
| Gradient Boosting |  48.2361  |  71.8303 | 0.992793 | 0.0229571   |       0.992011  |         0.996011  |

--------


## Model Selection

The tree-based methods and Gradient Boosting consistently outperformed the standard regression techniques. The Decision Tree model demonstrated had the best performance all around, perhaps there are some non-linear relationships which the classical regressors couldn't handle for.The inclusion of weather data resulted in a marginal improvement in model performance. The boost was slight,but we kept the weather variables in our final model, because it is relevant to the objective.


