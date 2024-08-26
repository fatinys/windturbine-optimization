# **Wind Turbine Capacity Optimzation**



## **Project Directory**

```
ENTER TREE HERE!!

```

## **Plan:**

We have two primary goals:
1. Create a model to predict the energy capacity for a turbine given its location.
2. Provide optimal zones in Texas that show potential for efficient wind farm projects.

### **Data Collection**: 
The data we need isnt all in the same place so we'll need to source it from several places. In `src/data`, youll find the relevant code that was used to create the data.

1. **Turbine Data**: In order to cut down on variables, we'll limit the search for turbines made within the last 10 years, this will keep the data relevant and consistent. For the sake of simplicity we'll choose not to consider manufacturer and model. We are getting this data from The U.S. Wind Turbine Database: https://eerscmap.usgs.gov/uswtdb/.
2. **Historical Wind Data**: For each turbine that we collect, using their respective coordinates. This data is coming from, https://open-meteo.com/. This data comes in daily spans, and weather data means we can't just simply average features over a 365 day span. We will attempt to use some crafty statistical methods to attempt to correctly characterize the data in context.

## Modeling

### Turbine Modeling

We will start by modelling only the features directly on the turbine dataset, lets see if we can accurately predict the rated capacity. If that goes well, we wouldn't even need to model based off of weather data.

--------

