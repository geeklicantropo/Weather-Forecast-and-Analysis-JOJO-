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
Run the command python3 Scripts/climate_prediction/main.py
