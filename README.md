# Water Consumption Forecasting Using Digital Twins and AI/ML

## Overview

This project focuses on forecasting water consumption using machine learning (ML) and artificial intelligence (AI) models, leveraging data from **Digital Twins** implemented for various villages in Spain. Digital Twins provide daily real-time and historical data for water consumption, allowing for accurate predictions and enhanced resource management.

We implemented a digital twin system that collects data from various sources, such as water meters, meteorology stations, and programmable logic controllers (PLCs). The data is then processed using AI/ML models to predict water consumption and detect leakages.

## Digital Twin Diagram

Below is a preview of the Digital Twin system diagram:

![Digital Twin Diagram](./images/Digital_Twins.png)

## Project Structure

- **Data Sources**:
  - **Historical Water Consumption Data**: Collected from sensors and databases over time.
  - **Real-time Water Consumption Data**: Continuously monitored and collected daily.
  - **Meteorological Data**: Sourced from meteorology stations to enhance the accuracy of the predictions.
  
- **Main Features**:
  - **Water Consumption Prediction**: AI and ML models, including **LSTM** and **Prophet**, are used for forecasting daily water usage.
  - **Leakage Detection**: Early detection of water leakages by analyzing consumption patterns.
  - **Energy Consumption and CO2 Footprint**: Monitoring the energy impact of water distribution and associated CO2 emissions.

- **Pre-Processing and AI/ML Pipeline**:
  The project follows a well-defined data pipeline that includes:
  - **Pre-processing Stage**: Data cleaning, normalization, and preparation for analysis.
  - **Analysis Stage**: Applying models like LSTM and Prophet for time series forecasting.
  - **Post-Processing Stage**: Interpretation of the model outputs to provide actionable insights regarding water consumption and leakage detection.

## Project Workflow

1. **Data Input**: Collection of historical and real-time water consumption data along with meteorological data.
2. **Pre-Processing**: Cleaning and preparation of the data for machine learning models.
3. **AI/ML Processing**: Applying LSTM and Prophet models to predict future water consumption and detect anomalies such as leakages.
4. **Output**: Predictions and analytics related to water usage, energy consumption, and environmental impact.

## Models Used

- **LSTM (Long Short-Term Memory)**: Used for time-series forecasting, particularly effective in handling sequential data.
- **Prophet**: A model designed for forecasting time series data, especially when the data exhibits strong seasonal effects.

## Technologies

- **Digital Twins**: Simulating the water consumption of villages in Spain, providing a virtual representation for better decision-making and forecasting.
- **AI/ML Techniques**: Leveraging advanced machine learning algorithms to generate accurate forecasts and detect anomalies in water usage.
  
## How to Run

1. **Set up the Environment**:
   - Install necessary dependencies by running:
     ```
     pip install -r requirements.txt
     ```

2. **Run the Jupyter Notebook**:
   - The core of the project is contained in the Jupyter notebook `Water_Consumption_Forecasting.ipynb`. Open it to explore the data and model workflow:
     ```
     jupyter notebook Water_Consumption_Forecasting.ipynb
     ```

## Future Work

- Expand the data sources to include additional villages.
- Improve the models' accuracy by incorporating more environmental and socioeconomic data.
- Integrate real-time alerts for water leakage and unusual consumption patterns.

## Contributors

- [Your Name]
  
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
