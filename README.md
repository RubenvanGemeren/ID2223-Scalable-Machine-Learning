# ID2223-Scalable-Machine-Learning

This assignment implements a air quality prediction service for the centre of Rotterdam using publicly available sensor and weather data using https://aqicn.org/api/ and https://open-meteo.com/. Using [Hopsworks](https://app.hopsworks.ai) to orchestrate and manage feature groups and prediction models. A Extreme boosted gradient model is used for predictions. The first image shows the predicted air quality for the next 7 days ([see image one](air_quality_prediction_service/ch03/docs/air_quality_model/assets/images/pm25_forecast.png)). The second image shows the predicted air quality with the actual air quality observed that day ([see image two](air_quality_prediction_service/ch03/docs/air_quality_model/assets/images/pm25_hindcast.png)).

The project consists of 4 notebooks:

## Backfill
- The first notebook sets up the project by creating the necessary API keys for Hopsworks, AQI, and Open Meteo. The main task of this notebook is to 'backfill' the historical data in our systems to be used for the creation of the model that will later be used for our predictions. The historical data is cleaned and prepared to be stored in Hopsworks as feature groups. This notebook is only run once to create an up-to-date backlog; future measurements will be processed and stored daily.

## Feature pipeline
- The next notebook is responsible for keeping our data up to date. This is done by running this particular notebook daily using GitHub Actions. The API keys created during the backfill are retrieved from Hopsworks and queried to get the latest weather and air quality data from our sources, updating the feature groups in the process.

## Training pipeline
- After creating and updating the datastore, we can build our predictor model. This is done by creating a Hopsworks Feature View and splitting our data into training and test data. The model used is called Extreme Gradient Boosting, a fast and flexible model great for achieving high predictive accuracy. After creating the model, it is stored in Hopsworks so it can be used later.

## Batch inference
- Finally, we use the created model and the data we have prepared to make actual air quality predictions. We do two things: first, we make air quality predictions for the next 7 days, and second, we compare today's air quality measurements with our predicted values to see how far off we were.
- As part of the bonus for this assignment, we implemented 'lagged air quality values' into our batch. These are the air quality values from the last three days that we add to every element in our batch, adding more context to the predictions. The problem we had to solve was how to account for lagged air quality values for weather forecast data in the future. No air quality values exist for days that have not yet occurred, so we had to create those ourselves. We used the same model to first fill out the lagged air quality values before using that augmented data to predict the actual air quality for our forecast. This way, we are able to add the air quality data as part of our features for future predictions.





# Published page
The final published page can be found here: https://rubenvangemeren.github.io/ID2223-Scalable-Machine-Learning/
