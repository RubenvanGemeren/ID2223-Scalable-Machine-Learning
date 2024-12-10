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


# Lab 2 - Fine tuning using Unsloth


### Checkpointing

We add checkpointing to the training by passing the "output_dir" parameter for the trainer. We checkpoint every 100 steps and only keep the latest checkpoint for storage efficiency.

### Fine tune model

The model has been trained approximately 1/2 an epoch (5600 steps) on the FineTome dataset. It has been trained both on Google Colab free and one of our personal GPUs.

### UI for model interaction

We have created a UI that lets users input ingredients and corresponding quantites (g). The model then composes a meal using the ingredients and gives the user the instructions of how to make the meal.

The UI is available: (https://huggingface.co/spaces/ID2223JR/lab2)

### Improvements

- Tuning hyperparameters:
1. Could add dropout > 0 to prevent overfitting. According to Unsloth documentation they only support dropout=0 for now, as they have optimized the dropout internally somehow.
2. The "r" parameter, which is the rank of the low-rank decomposition for factorizing weight matrices, could also be tuned. A higher value would retain more information but require more computational resources. A lower value would mean fewer parameters but more efficient training, and if too small a potential risk for performance drop.
3. The "lora_alpha" parameter, which is the scaling factor for the low-rank matrices' contribution could also be tuned. Higher value incerases the influence of the matrices', speeds up convergence but increases a risk for overfitting. Lower value decrease influence, meaning it would be requried to train the model for more steps.

We havn't experimented with these features, as we feel we didn't have the computational resources to do so. Instead we tried fine tuning using another dataset.

- Identifying new data sources to train a better model for our purpose:
1. We have identified a dataset perfect for the use of our model: (https://huggingface.co/datasets/mbien/recipe_nlg)
2. This dataset contains over 2 million recipes and directions for cooking the meals.
3. We created a second notebook that downloads this dataset, then parses it into the correct format (instruction-response format) to be able to use it for fine tuning the model further.
4. We weren't able to complete a full epoch of training, since the dataset is huge. In the end, we managed to run it 30.000 steps (10% of an epoch for that dataset).


