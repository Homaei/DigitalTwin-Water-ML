# Water Consumption Forecasting Using Digital Twins and AI/ML
## Digital Twin in Water Industries
![Digital Twin](src/DT%20in%20Water%20industries.webp)

## Overview

This project leverages **Digital Twin** technology, advanced **Machine Learning (ML)**, and **Artificial Intelligence (AI)** models to accurately forecast water consumption across multiple rural villages in Spain. We aim to support effective resource management, optimize infrastructure planning, and enhance sustainability by integrating real-time and historical data Digital Twins collects.



### Digital Twin System

Our **Digital Twin** setup collects data from multiple sources:

- **Water Meters**: Capture regular water usage data.
- **Meteorological Stations**: Provide weather data, helping identify correlations with water consumption.
- **Programmable Logic Controllers (PLCs)**: Monitor and control the water distribution system for improved operational efficiency.

This data is aggregated and processed through ML and AI models to forecast water consumption and detect anomalies. By identifying unusual patterns early, we enable timely responses to leaks, reduce waste, and support efficient water usage.

### Importance of Water Consumption Forecasting 

Accurate water forecasting is crucial for managing finite water resources, supporting infrastructure planning, and enhancing sustainability. The benefits of a robust forecasting model include:

1. **Improved Resource Management**: Predictions allow for optimized supply, minimizing waste and cost.
2. **Better Infrastructure Decisions**: Insights from forecasts support maintenance and expansion prioritization.
3. **Enhanced Sustainability**: By aligning distribution with demand, we conserve energy and reduce environmental impacts.

### Forecasting Models and Methodologies

The project evaluates several models for water consumption forecasting over 6-month and 18-month horizons, comparing traditional and advanced ML methods:

1. **Prophet Model**: A time series forecasting model designed for seasonality, holidays, and custom regressors, tailored for periodic consumption changes.
2. **XGBoost and LightGBM**: Efficient and accurate boosted tree algorithms, further improved with custom feature engineering.
3. **LSTM Neural Networks**: Long Short-Term Memory networks, ideal for capturing long-sequence dependencies essential in forecasting water trends.

### Feature Engineering

Advanced feature engineering techniques were applied to enhance predictive accuracy:

- **Lag Features**: Integrate past consumption data to capture temporal patterns.
- **Rolling Statistics**: Use rolling means, standard deviations, and max values to smooth out short-term fluctuations.
- **Domain-Specific Variables**: Incorporate factors like maximum daily temperature and day of the week, significantly impacting water usage.

### Hyperparameter Tuning

We fine-tuned each model to optimize performance:

- **Prophet Model**: Tuned seasonality, added holiday effects, and custom regressors.
- **LSTM Model**: Adjusted dropout rates, learning rates, sequence length, and layer units through grid and randomized search.
- **Stacking Ensemble Methods**: Combined outputs from XGBoost and LightGBM models to increase robustness by capturing different aspects of the data.



## Digital Twin Diagram

Below is a preview of the Digital Twin system diagram:

![Digital Twin Diagram](./Digital_Twins.png)

## Project Structure

- **Data Sources**:
  - **Historical Water Consumption Data**: Collected from sensors and databases over time.
![image](https://github.com/user-attachments/assets/1aca6b3a-c02e-4af3-8424-8c388523554e)

    
  - **Real-time Water Consumption Data**: Continuously monitored and collected daily. This data is captured every 8 hours from water meters in the village. 
![image](https://github.com/user-attachments/assets/ffd428b4-6ce2-424d-ac28-13b9f2f56db5)

  - **Meteorological Data**: Collected from meteorological stations to improve prediction accuracy. We evaluated the relationship between different parameters and water consumption using the Pearson correlation method. Our analysis revealed that maximum temperature positively correlates with water consumption. The results of this correlation analysis are presented in the following figure.
![image](https://github.com/user-attachments/assets/36b6d5cd-4fb2-47a8-92ab-b8370d4297e0)
![image](https://github.com/user-attachments/assets/ff28496e-721b-4df8-b719-55f2f1782ec0)

  
- **Main Features**:
  - **Water Consumption Prediction**: AI and ML models, including **LSTM** and **Prophet**, are used for forecasting daily water usage.
  - **Leakage Detection**: Early detection of water leakages by analyzing consumption patterns.
  - **Energy Consumption and CO2 Footprint**: Monitoring the energy impact of water distribution and associated CO2 emissions (These parameters result from maintaining the water distribution network). For example, each operator has several tasks with a variety of time, location, priority, and other metrics, and the scheduling with preemption (Urgent tasks) is an NP-hard problem.




- **Pre-Processing and AI/ML Pipeline**:
  The project follows a well-defined data pipeline that includes:
  - **Pre-processing Stage**: Data cleaning, normalization, and preparation for analysis.
  - **Analysis Stage**: Applying models like LSTM and Prophet for time series forecasting.
  - **Post-Processing Stage**: Interpret the model outputs to provide actionable insights regarding water consumption and leakage detection.

## Project Workflow

1. **Data Input**: Collecting historical and real-time water consumption and meteorological data.
2. **Pre-Processing**: Cleaning and preparing the data for machine learning models.
3. **AI/ML Processing**: Applying LSTM and Prophet models to predict future water consumption and detect anomalies such as leakages.
4. **Output**: Predictions and analytics related to water usage, energy consumption, and environmental impact.

## Models Used

| No | Main Method | Algorithm Name                   | Differences/Parameters                                                                 |
|----|-------------|----------------------------------|----------------------------------------------------------------------------------------|
| 1  | Prophet     | Prophet Basic                    | Basic model, no additional seasonality or regressors                                   |
|    |             | Prophet + Seasonality            | Includes seasonality components (e.g., yearly or weekly)                               |
|    |             | Advanced Prophet                 | Includes advanced features like holidays added regressors                              |
|    |             | Prophet Adv. Engineering         | Custom feature engineering (lag, rolling means, etc.)                                  |
| 2  |  LSTM       | LSTM Basic                       | Vanilla LSTM, no additional tuning or feature engineering                              |
|    |             | LSTM Hyperparameter Tuning       | LSTM with tuned hyperparameters (e.g., learning rate, units)                           |
|    |             | LSTM + GRU Hybrid                | Combination of LSTM and GRU layers for better generalization                           |
|    |             | LSTM Rolling Mean Features       | LSTM with rolling mean features for smoother predictions                               |
| 3  | XGBoost     | XGBoost Basic                    | Basic XGBoost model without additional feature engineering                             |
|    |             | XGBoost with Feature Engineering | XGBoost with advanced feature engineering (lag, etc.)                                  |
| 4  | LightGBM    | LightGBM Basic                   | Basic LightGBM model, no feature engineering                                           |
|    |             | LightGBM with Feature Engineering| LightGBM with engineered features (e.g., lags, moving averages)                        |
| 5  | Stacking    | Stacking XGBoost + LightGBM      | Ensemble of XGBoost and LightGBM, stacking the models                                  |
------------------------------------------------------------------------------------------------------------------------------------------------

### Model Evaluation Metrics

Each model was evaluated on the following metrics to identify top performers across 6-month and 18-month forecasting periods:

1. **Mean Absolute Error (MAE)**: Average error magnitude in predictions.
2. **Root Mean Squared Error (RMSE)**: Provides a higher penalty for large prediction errors.
3. **Mean Absolute Percentage Error (MAPE)**: Standardized error measurement as a percentage for cross-model comparison.

Based on these metrics, the best models for each forecasting period are presented in the following sections.



| No | Model                                                                                          | 6 M MAE | 6 M RMSE | 6 M MAPE | 18 M MAE | 18 M RMSE | 18 M MAPE |
|----|------------------------------------------------------------------------------------------------|---------|----------|----------|----------|-----------|-----------|
| 1   | LightGBM                                                                                        | 5.90    | 8.25     | 19.64%   | 11.77    | 18.31     | 24.98%    |
| 2   | LSTM Hyper. Tuning Plus                                                                         | 5.96    | 9.38     | 18.64%   | 12.63    | 20.66     | 25.61%    |
| 3   | Prophet Adv. Engineering                                                                        | 6.21    | 8.75     | 20.61%   | 10.12    | 17.02     | 21.43%    |
| 4   | Advanced Prophet                                                                                | 6.24    | 8.78     | 20.77%   | 11.14    | 18.02     | 22.34%    |
| 5   | LSTM Rolling Mean Features                                                                      | 7.94    | 10.82    | 27.59%   | 12.33    | 20.57     | 24.67%    |

These results underscore the effectiveness of combining feature engineering, hyperparameter tuning, and advanced machine learning techniques to improve water consumption forecasting accuracy. The models also demonstrated robust performance in the 18-month forecasts, showcasing their versatility across different forecasting horizons.


| No | Model                                                                                          | 6 M MAE | 6 M RMSE | 6 M MAPE | 18 M MAE | 18 M RMSE | 18 M MAPE |
|----|------------------------------------------------------------------------------------------------|---------|----------|----------|----------|-----------|-----------|
| 1  | Prophet Basic                                                                                  | 10.37   | 13.66    | 22.45%   | 19.70    | 28.68     | 22.45%    |
| 2  | Prophet + Seasonality                                                                           | 12.39   | 14.91    | 22.45%   | 24.25    | 35.02     | 22.45%    |
| 3  | Advanced Prophet                                                                                | **6.24**    | **8.78**     | **20.77%**   | **11.14**    | **18.02**     | **22.34%**    |
| 4  | **Prophet Adv. Engineering**                                                                    | **6.21**| **8.75** | **20.61%**| **10.12**| **17.02** | **21.43%**|
| 5  | XGBoost                                                                                         | 7.02    | 8.74     | 24.93%   | 12.34    | 18.50     | 27.49%    |
| 6  | **LightGBM**                                                                                    | **5.90**| **8.25** | **19.64%**| **11.77**| **18.31**     | **24.98%**    |
| 7  | Stacking XGBoost + LightGBM                                                                     | 6.57    | 8.70     | 22.45%   | 12.48    | 18.94     | 27.62%    |
| 8  | LSTM Network                                                                                    | 7.31    | 10.74    | 22.43%   | 16.03    | 22.16     | 39.95%    |
| 9  | LSTM Hyperparameter Tuning                                                                      | 6.51    | 9.61     | 21.34%   | 12.72    | 20.61     | 26.47%    |
| 10 | **LSTM Hyper. Tuning Plus**                                                                     | **5.96**| **9.38** | **18.64%**| **12.63**| **20.66**     | **25.61%**    |
| 11 | LSTM Hyper. Tuning Changed Params                                                               | 6.52    | 9.63     | 21.38%   | 13.33    | 20.60     | 29.21%    |
| 12 | LSTM + GRU Hybrid                                                                               | 8.18    | 10.10    | 30.10%   | 14.64    | 22.32     | 34.06%    |
| 13 | **LSTM Rolling Mean Features**                                                                  | **7.94**| **10.82**    | **27.59%**| **12.33**| **20.57**     | **24.67%**    |




Prophet model's results:

![FIG10-A-Prophet Model with Advanced Feature Engineering 6m](https://github.com/user-attachments/assets/79376d28-8b70-4b35-af8b-a9557cb700d8)
Figure 4:  Prophet Model with Advanced Feature Engineering 6 months forecasting

![FIG10-B-Prophet Model with Advanced Feature Engineering 18m](https://github.com/user-attachments/assets/f2d7c49e-13db-4428-aa2b-9526ff639bf5)
Figure 5:  Prophet Model with Advanced Feature Engineering 18 months forecasting


# Water Distribution System (WDS) Maintenance Optimization

## Overview

In rural Water Distribution Networks (WDNs), operators face the challenge of efficiently routing and scheduling maintenance tasks across vast areas with varied priorities and dependencies. These tasks often rely on operators’ judgment, which can lead to inefficiencies, especially when handling simultaneous tasks with differing priorities. This project addresses these challenges by developing a systematic approach to optimize routing and scheduling, thereby reducing operational costs, such as travel time, fuel consumption, and CO₂ emissions.

### Problem Description

The maintenance scheduling in WDNs is modeled as a complex **Single Machine Scheduling** problem with preemptive tasks, variable release times, and task dependencies. This problem is NP-hard and involves several constraints, including:

- **Task Prioritization**: Tasks are assigned different priorities, requiring high-priority tasks to be addressed promptly.
- **Emergency Tasks**: High-priority emergency tasks can arrive at any time and need to be incorporated into the existing schedule, often with penalties for delays.
- **Task Dependencies**: Some tasks are dependent on the completion of others, which must be respected to avoid operational conflicts.

The objective is to develop an optimized schedule that minimizes completion time, fuel consumption, CO₂ emissions, and task delays, while ensuring efficient handling of tasks and respecting dependencies and preemptions.

### Mathematical Model and Objectives

To solve this problem, we formulated a **Constraint Programming (CP) model** that accounts for deterministic parameters, aiming to:

1. **Minimize Total Completion Time** $$(C_{\text{max}})$$
2. **Minimize Total Fuel Consumption** $$(F_{\text{total}})$$
3. **Minimize Total CO₂ Emissions** $$(C_{\text{total}})$$
4. **Minimize Delays and Penalties** for high-priority tasks $$(D_{\text{total}})$$

### Model Components

The model includes various sets, indices, parameters, decision variables, and constraints:

- **Sets and Indices**:
  - $$T$$: Set of all tasks
  - $$D$$: Set of task dependencies, where a dependency $$(i, j)$$ indicates task $$j$$ must follow task $$i$$

- **Parameters**:
  - $$p_i$$: Processing time for each task $$i$$
  - $$d_{ij}$$: Travel time between tasks $$i$$ and $$j$$
  - $$f_i$$: Fuel consumption for each task $$i$$
  - $$c_i$$: CO₂ emissions for each task $$i$$
  - $$r_i$$: Release time for task $$i$$ (emergency tasks have $$r_i \geq 0$$)

- **Decision Variables**:
  - **Task Scheduling Variables**: Define start and end times for each task or segment.
  - **Sequencing Variables**: Determine the order in which tasks are performed.
  - **Auxiliary Variables**: Help manage task preemptions and dependencies.

### Objective Function

The overall objective function is a weighted sum of the various components we aim to minimize:

$$
\min Z = w_t \times (C_{\text{max}} - S) + w_f \times F_{\text{total}} + w_c \times C_{\text{total}} + w_d \times D_{\text{total}}
$$

where:
- $$w_t, w_f, w_c, w_d$$ are the weights for completion time, fuel, CO₂ emissions, and delays.
- $$C_{\text{max}}$$: Completion time of the last task.
- $$F_{\text{total}}$$: Total fuel consumed.
- $$C_{\text{total}}$$: Total CO₂ emissions.
- $$D_{\text{total}}$$: Total delays for emergency tasks.

### Constraints

To ensure the model functions effectively within operational limits, we included several constraints:

1. **Processing Time**: Total scheduled processing time for each task matches its required time.
2. **Precedence for Dependencies**: If task $$j$$ depends on task $$i$$, task $$j$$ cannot start until task $$i$$ completes.
3. **Non-Overlap Constraint**: Ensures no overlap in tasks on a single machine.
4. **Work Hours**: Tasks are scheduled within defined working hours.
5. **Emergency Task Release Times**: Emergency tasks cannot start before their release time.
6. **Limit on Preemptions**: Each task has a maximum number of preemptions.

### Optimization Approach

Given the NP-hard nature of the problem, **Constraint Programming (CP)** was chosen for its effectiveness in solving complex scheduling problems with various dependencies and constraints. We used a CP solver (e.g., Google OR-Tools) to identify the optimal scheduling arrangement.

### Performance Comparison

The optimized model was compared against conventional operator methods, demonstrating significant improvements:

| Metric                             | Conventional Method | Proposed Model | Improvement (%) |
|------------------------------------|---------------------|----------------|-----------------|
| Total Completion Time              | 180.58 hours       | 155.24 hours   | 14%            |
| Delays and Penalties               | 17.5 hours         | 13.15 hours    | 25%            |
| CO₂ Emissions                      | 660.8 kg           | 545.7 kg       | 17%            |
| Fuel Consumption                   | 85.58 Litres       | 71.98 Litres   | 16%            |
| Efficiency and Utilization         | 86.17%             | 92.23%         | 7%             |

These improvements illustrate the effectiveness of our model in reducing operational costs and environmental impact, while ensuring timely and efficient task completion.

### Visual Analysis

 ![image](https://github.com/user-attachments/assets/2139cd4d-e107-4197-abf5-494f3db7bacb)

 
Below are visual representations of key metrics analyzed in the model:

![Completion Time](src/Completion%20Time.png)
![Delay and Penalties](src/Delay%20and%20Penalties.png)
![Fuel Consumption](src/Fuel%20Consumption.png)
![CO₂ Emissions](src/CO2%20Emissions.png)
![Efficiency and Utilization](src/Efficiency%20and%20Utilization.png)

---



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
   - The project's core is in the Jupyter notebook `Water_Consumption_Forecasting.ipynb`. Open it to explore the data and model workflow:
     ```
     jupyter notebook Water_Consumption_Forecasting.ipynb
     ```

## Future Work

- Expand the data sources to include additional villages.
- Improve the models' accuracy by incorporating more environmental and socioeconomic data.
- Integrate real-time alerts for water leakage and unusual consumption patterns.

## Contributors

- [Hubert Homaei](https://github.com/homaei), [Oscar Mogollon](https://github.com/omogollo2). 


## How to cite to this research:

📌 APA (7th Edition)
arXiv version:

Homaei, M., Di Bartolo, A. J., Ávila, M., Mogollón-Gutiérrez, Ó., & Caro, A. (2024). Digital transformation in the water distribution system based on the digital twins concept. arXiv. https://doi.org/10.48550/arXiv.2412.06694

MDPI Preprints version:

Homaei, M., Di Bartolo, A. J., Ávila, M., Mogollón-Gutiérrez, Ó., & Caro, A. (2024). Digital transformation in the water distribution system based on the digital twins concept. MDPI Preprints. https://doi.org/10.20944/preprints202412.0756.v1

📌 IEEE
arXiv version:

[1] M. Homaei, A. J. Di Bartolo, M. Ávila, Ó. Mogollón-Gutiérrez, and A. Caro, “Digital transformation in the water distribution system based on the digital twins concept,” arXiv, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2412.06694.

MDPI Preprints version:

[2] M. Homaei, A. J. Di Bartolo, M. Ávila, Ó. Mogollón-Gutiérrez, and A. Caro, “Digital transformation in the water distribution system based on the digital twins concept,” MDPI Preprints, Dec. 2024. [Online]. Available: https://doi.org/10.20944/preprints202412.0756.v1.

📌 Chicago (Author-Date)
arXiv version:

Homaei, MohammadHossein, Agustín Javier Di Bartolo, Mar Ávila, Óscar Mogollón-Gutiérrez, and Andrés Caro. 2024. “Digital Transformation in the Water Distribution System Based on the Digital Twins Concept.” arXiv. https://doi.org/10.48550/arXiv.2412.06694.

MDPI Preprints version:

Homaei, MohammadHossein, Agustín Javier Di Bartolo, Mar Ávila, Óscar Mogollón-Gutiérrez, and Andrés Caro. 2024. “Digital Transformation in the Water Distribution System Based on the Digital Twins Concept.” MDPI Preprints, December. https://doi.org/10.20944/preprints202412.0756.v1.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
