# Forex Hedging Deep Learning Program

## Overview
This project is an advanced forex hedging recommendation system that leverages deep learning techniques to provide personalized hedging strategies. It employs an ensemble of neural network models, including LSTM, GRU, and CNN, to analyze historical forex data and user inputs for generating optimal hedging recommendations.

## Features
Supports multiple currency pairs worldwide
Utilizes yfinance for real-time forex data retrieval
Implements Bayesian optimization for hyperparameter tuning
Incorporates multiple technical indicators (RSI, MACD, Bollinger Bands, Stochastic Oscillator, ATR)
Combines neural network predictions with rule-based recommendations
Provides detailed implementation steps for each recommended hedging strategy
Generates personalized recommendations based on user inputs and risk profile

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- SciPy
- yfinance
- scikit-optimize

## Installation
Clone this repository: git clone [repository URL]
Install the required packages:
pip install torch numpy pandas scikit-learn scipy yfinance scikit-optimize

## Usage
Run the main script:
Copypython main.py
You will be prompted to enter various details about your forex exposure and company profile, including:
1. Currency pair (e.g., USD/EUR)
2. Amount of exposure
3. Duration of exposure
4. Risk tolerance
5. Company size and resources
6. Hedging preferences and constraints

The program will then:
1. Retrieve historical forex data
2. Train and optimize the ensemble model
3. Generate hedging recommendations
4. Provide detailed implementation steps for top recommendations

## How it Works
1. **Data Retrieval**: Uses yfinance to download historical forex data.
2. **Data Preparation**: Calculates technical indicators and prepares sequences for the neural network models.
3. **Model Training**: Employs an ensemble of LSTM, GRU, and CNN models, optimized using Bayesian optimization.
4. **Recommendation Generation**: Combines neural network predictions with rule-based recommendations.
5. **Strategy Implementation**: Provides detailed steps for implementing each recommended hedging strategy.

## Key Components
1. **load_and_preprocess_data()**: Prepares and engineers features from raw forex data.
2. **EnsembleModel**: Combined LSTM, GRU, and CNN neural network model.
3. **optimize_hyperparameters()**: Uses Bayesian optimization for hyperparameter tuning.
4. **neural_network_recommendation()**: Generates recommendations based on the trained model.
5. **recommend_hedging_methods()**: Provides rule-based recommendations.
6. **get_hedging_template()**: Generates detailed implementation steps for each strategy.

## Limitations
Forex markets are inherently unpredictable and affected by many external factors. This tool should be used for educational and informational purposes only, not for actual trading decisions.
The accuracy of recommendations can vary significantly depending on market conditions and the quality of input data.

## Disclaimer
This software is for educational and research purposes only. Do not use it to make any type of investment or hedging decisions. Always consult with a qualified financial advisor before making financial decisions. The creators of this project are not responsible for any financial losses incurred from using this tool.
