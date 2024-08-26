# **Wind Turbine Capacity Optimzation**

## **CURRENT BUILD**
Currently, we have the collection, processing, and modelling done. All that is left is to deploy the model via Flask locally, and then hopefully bring it to production with Azure. Working on an app that will allow a user to run the model locally.
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




--------

