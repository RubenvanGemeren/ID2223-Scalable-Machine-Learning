# ID2223-Scalable-Machine-Learning

# Lab 1 - Air Quality Prediction Service

This assignment implements a air quality prediction service for the centre of Rotterdam using publicly available sensor and weather data using https://aqicn.org/api/ and https://open-meteo.com/. Using [Hopsworks](https://app.hopsworks.ai) to orchestrate and manage feature groups and prediction models. A Extreme boosted gradient model is used for predictions. The first image shows the predicted air quality for the next 7 days ([see image one](air_quality_prediction_service/ch03/docs/air_quality_model/assets/images/pm25_forecast.png)). The second image shows the predicted air quality with the actual air quality observed that day ([see image two](air_quality_prediction_service/ch03/docs/air_quality_model/assets/images/pm25_hindcast.png)).

Lab 1 consists of 4 notebooks:

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

### Uploading model

The model is uploaded to our Hugging Face organization (https://huggingface.co/ID2223JR) in different formats (to be able to run with and without GPU).

### UI for model interaction

We have created a UI that lets users input ingredients and corresponding quantites (g). The model then composes a meal using the ingredients and gives the user the instructions of how to make the meal.

The user can choose between two models, the original model fine-tuned on the FineTome (https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset and the model fine-tuned on the RecipeNLG (https://huggingface.co/datasets/mbien/recipe_nlg) dataset.

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

# Project - Fantasy Premier League Points Predictor

## Introduction
The goal of this project is to predict the points that Fantasy Premier League (FPL) players will score in the upcoming game week. This will help FPL managers make informed decisions about transfers, captaincy, and team selection. The data sources include the Fantasy Premier League’s official API, which provides player statistics, fixture data, and historical performance metrics. The model is built using a backfill of all the data in a season and utilizes current live data to predict points.

## Problem Description
Predicting player performance, both in real-life and fantasy scenarios, is highly beneficial. As of the 2024 season, the estimated value of the first-place prize in FPL is approximately €13,500. The world of football is increasingly adopting artificial intelligence, with more than a billion euros being invested in developing home-grown talent. Predictive services are becoming essential.

The biggest challenge when predicting player performance is the inherent uncertainty in professional sports. Player performance can vary dramatically due to injuries, confidence, personal challenges, and more. In an ideal world, a prediction model would incorporate data from multiple sources to create a comprehensive context for player performance. While this is a school project with limited time and resources, we aim to achieve the most accurate predictions possible within these constraints.

## Tools
This project is implemented in Python, leveraging PyTorch, Scikit-learn, and pandas for data processing and feature engineering. Hopsworks is used for storing features and predictions. For deployment, we use [Hugging Face Spaces](https://huggingface.co/spaces/ID2223JR/fpl_app).

## Data
The data for this project is fetched from the [Fantasy Premier League API](https://fantasy.premierleague.com/api). This API provides information about players, teams, and matches in the English Premier League. The dataset is updated daily and contains metrics such as goals scored, assists, clean sheets, and other performance indicators. These metrics are used to predict the points players will receive in a game week. Points are determined by various actions, such as playtime, goals (differentiated by position), avoiding penalties, and more. Negative points can also be awarded for yellow or red cards.

Additionally, a static dataset provides supplementary information such as team names and player positions. This dataset is not part of the prediction pipeline but aids in feature interpretation.

## Methodology and Algorithm
The pipeline structure is inspired by the first lab assignment, which involved a serverless ML pipeline to predict air quality values. While the project structure is similar, the implementation diverges significantly.

### 1. Backfill
The historical data used to train the model is cleaned and prepared for storage in Hopsworks as feature groups. This notebook is run once to create an up-to-date backlog. Future measurements are processed and stored daily. A crucial aspect of feature engineering is shifting the data by one week, which enables predictions.

### 2. Feature Pipeline
This notebook updates the dataset daily using GitHub Actions. API keys created during the backfill process are retrieved from Hopsworks and used to fetch the latest data from our sources, updating the feature groups accordingly.

### 3. Training Pipeline
Once the datastore is ready, we build the prediction model. A Hopsworks Feature View is created, and the data is split into training and testing sets. We use Extreme Gradient Boosting (XGBoost), a fast and flexible model well-suited for achieving high predictive accuracy. The trained model is then stored in Hopsworks for later use.

### 4. Batch Inference
Finally, the trained model and prepared data are used to make predictions for the next game week. Predictions are updated for the latest game week by incorporating the actual points scored by players.

![Pipeline Diagram](a414853e7d0a62093a9a681fe7ec1350319da28c.png)

## Results
The user interface for this project is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/ID2223JR/fpl_app). It displays all active players, their predicted scores, and their historical scores for comparison. Interestingly, some players receive negative points due to fouls, which is accurately reflected in the predictions.

When comparing the top-rated players predicted by our model with other sources such as the [Premier League Official Site](https://www.premierleague.com/stats/top/players) or [Goal.com](https://www.goal.com/en/lists/premier-league-player-of-the-season-2024-25-power-rankings/blt350cdd828461eaeb), we observe significant similarities.

### Model Evaluation
The final model achieved the following metrics:
- **Mean Squared Error (MSE):** 6.15
- **Root Mean Squared Error (RMSE):** 2.48
- **R-Squared (ℓ²):** 0.093

An RMSE of 2.48 indicates that predictions will generally fall within 2.48 points of the actual score. Most player points range between -1 and 18, suggesting that the model captures general trends but struggles with finer distinctions among players of similar caliber.

The R-squared value of 0.093 indicates that approximately 9% of the variability in player points is explained by the model, while 91% is due to external factors not captured in the dataset. This underscores the limitations of the features used in the model.

## Discussion
Predicting player points involves numerous variables, many of which are not included in the dataset. Factors such as injuries, opponent strength, and match context significantly influence performance but are not represented in the provided data.

Our model performs reasonably well given the available data. Improvements would require additional features and more robust datasets. For example, FPL assigns points differently based on player positions (e.g., strikers receive fewer points for goals than defenders). This positional context is not explicitly captured in our data, leading to a more generalized model.

Future enhancements could include experimenting with more advanced models such as LightGBM, CatBoost, or neural networks. However, time constraints prevented their implementation in this project.

## Conclusion
This project provided valuable insights into building predictive systems. We learned that achieving high accuracy requires extensive and diverse data. While using Hopsworks offered many advantages, it also presented challenges when issues arose.

For future work, we recommend testing additional models and incorporating more data sources to enrich the feature set. There is significant potential for refining the existing pipeline to achieve better results.