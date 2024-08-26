# **Wind Turbine Capacity Optimzation**

## **CURRENT BUILD**
Currently, we have the collection, processing, and modelling done. All that is left is to deploy the model via Flask locally, and then hopefully bring it to production with Azure. Working on an app that will allow a user to run the model locally. app.py is currently not functional.
--------------------------

## **Project Directory**

```
ENTER TREE HERE!!

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

From above, the standard regression methods seemed to be clearly beat out by the tree-based methods and Gradient Boosting. The Decision Tree model wins out on all parameters, adding the weather data added a slight performance boost, but we are keeping the weather data due to the nature of the core objective of the project.