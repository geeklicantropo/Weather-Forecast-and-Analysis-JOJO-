# Weather-Forecast-and-Analysis-JOJO-
Automatization of weather forecast and analysis for Brazilian regions

# Requirements
* Python 3.10
* CUDA 12.X
* Run pip install -r requirements.txt to install other requirements

# How to collect the data?
In the root folder ( where this README is located), run the following files in order:
* python3 Scripts/extracting_data.py (and wait for it to finish)
* python3 Scripts/treat_data.py (and wait for it to finish)
* python3 Scripts/data_concatenation.py (and wait for it to finish)

# How to make predictions?
1. First run data splitting:
* Run the command python3 Scripts/climate_prediction/src/data_processing/split_data.py

2. Then run data processing:
* Run the command python3 Scripts/climate_prediction/process_data.py

3. Set up the configurations:
Run the comamnd python3 Scripts/climate_prediction/setup_config.py

4. Only then, run the predictions:
* Run the command python3 Scripts/climate_prediction/main.py
