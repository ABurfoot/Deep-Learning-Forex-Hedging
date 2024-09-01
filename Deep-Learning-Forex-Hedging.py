import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from scipy import stats

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defines the list of available hedging methods
# These methods will be considered by both the neural network and rule-based systems
HEDGING_METHODS = [
    "Forward Contracts",
    "Currency Futures",
    "Currency Options",
    "Money Market Hedges",
    "Natural Hedging",
    "Currency Diversification",
    "Dynamic Hedging",
    "Spot Market",
    "Currency Swaps",
    "Cross-Currency Swaps",
    "Proxy Hedging",
    "Netting",
    "Matching",
    "Leading and Lagging",
    "Participating Forwards",
    "Range Forwards"
]

# Custom attention mechanism to focus on relevant parts of the input sequence which assists the model in identifying which time steps are most important for prediction
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, user_feature_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size + user_feature_size, 1)
        
    def forward(self, lstm_output, user_features):
        user_features_expanded = user_features.unsqueeze(1).expand(-1, lstm_output.size(1), -1)
        combined = torch.cat((lstm_output, user_features_expanded), dim=2)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

# LSTM model with attention mechanism for improved feature extraction
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, user_feature_size):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_size, user_feature_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, user_features):
        lstm_output, _ = self.lstm(x)
        attn_output = self.attention(lstm_output, user_features)
        output = self.fc(attn_output)
        return output

# Gated Recurrent Unit (GRU) model for sequence processing
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Convolutional Neural Network (CNN) model for local feature extraction
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = self.fc(x)
        return x

# Feature interaction layer captures interactions between market features and user features
class FeatureInteractionLayer(nn.Module):
    def __init__(self, market_feature_size, user_feature_size, output_size):
        super(FeatureInteractionLayer, self).__init__()
        self.market_projection = nn.Linear(market_feature_size, output_size)
        self.user_projection = nn.Linear(user_feature_size, output_size)
        self.interaction = nn.Bilinear(output_size, output_size, output_size)
        
    def forward(self, market_features, user_features):
        market_proj = self.market_projection(market_features)
        user_proj = self.user_projection(user_features)
        return self.interaction(market_proj, user_proj)

# Ensemble model combining LSTM, GRU, and CNN, which leverages the strengths of all these neural network architectures for improved predictions
class EnsembleModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_user_features, dropout_rate=0.3):
        super(EnsembleModel, self).__init__()
        self.lstm = LSTMWithAttention(input_size, hidden_size, num_layers, hidden_size, num_user_features)
        self.gru = GRUModel(input_size, hidden_size, num_layers, hidden_size)
        self.cnn = CNNModel(input_size, hidden_size)
        self.user_fc = nn.Linear(num_user_features, hidden_size)
        self.feature_interaction = FeatureInteractionLayer(hidden_size, hidden_size, hidden_size)
        self.final_fc = nn.Linear(hidden_size * 5, len(HEDGING_METHODS))
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 5)
    
    def forward(self, x, user_features):
        lstm_out = self.lstm(x, user_features)
        gru_out = self.gru(x)
        cnn_out = self.cnn(x)
        user_out = self.user_fc(user_features)
        interacted_features = self.feature_interaction(lstm_out, user_out)
        combined = torch.cat((lstm_out, gru_out, cnn_out, user_out, interacted_features), dim=1)
        combined = self.dropout(combined)
        
        # Conditional batch normalization
        if combined.size(0) > 1:
            combined = self.batch_norm(combined)
        
        return torch.sigmoid(self.final_fc(combined))

# Collect user input about user's forex hedging needs and preferences which is used to personalize the hedging recommendations
def get_user_input():
    print("Welcome to the Optimized Forex Hedging Recommendation System")

    trade_info = {}

    def get_validated_input(prompt, input_type, valid_range=None, valid_options=None):
        while True:
            try:
                user_input = input(prompt)
                if input_type == bool:
                    if user_input.lower() not in ['yes', 'no']:
                        raise ValueError("Please enter 'yes' or 'no'.")
                    return user_input.lower() == 'yes'
                elif input_type == float:
                    value = float(user_input)
                    if valid_range and (value < valid_range[0] or value > valid_range[1]):
                        raise ValueError(f"Please enter a number between {valid_range[0]} and {valid_range[1]}.")
                    return value
                elif input_type == str:
                    if valid_options and user_input.lower() not in valid_options:
                        raise ValueError(f"Please enter one of: {', '.join(valid_options)}")
                    return user_input.lower()
                else:
                    return input_type(user_input)
            except ValueError as e:
                print(f"Invalid input: {e}")

    trade_info['trade_type'] = get_validated_input("Are you buying or selling foreign currency? (buy/sell): ", str, valid_options=['buy', 'sell'])
    trade_info['currency_pair'] = input("Enter the currency pair (e.g., USD/EUR): ").upper()
    trade_info['amount'] = get_validated_input("Enter the amount of base currency: ", float, valid_range=(0, float('inf')))
    trade_info['duration'] = get_validated_input("Enter the duration of your exposure in days: ", int, valid_range=(1, 3650))

    print("\nRisk and Volatility Assessment:")
    trade_info['risk_tolerance'] = get_validated_input("On a scale of 0-100, how much financial risk can your company tolerate? (0 = very low, 100 = very high): ", float, valid_range=(0, 100))
    trade_info['volatility_appetite'] = get_validated_input("On a scale of 0-100, how would you characterize your company's appetite for currency value fluctuations? (0 = very conservative, 100 = very aggressive): ", float, valid_range=(0, 100))

    print("\nCompany Information:")
    employees = get_validated_input("How many employees does your company have? ", int, valid_range=(1, 1000000))
    if employees <= 10:
        trade_info['company_size'] = 'micro'
    elif employees <= 49:
        trade_info['company_size'] = 'small'
    elif employees <= 249:
        trade_info['company_size'] = 'medium'
    else:
        trade_info['company_size'] = 'large'

    print("\nHedging Experience and Resources:")
    trade_info['hedging_experience'] = get_validated_input("On a scale of 0-100, how experienced is your company with forex hedging? (0 = least, 100 = most): ", float, valid_range=(0, 100))
    trade_info['has_treasury_team'] = get_validated_input("Do you have a dedicated treasury team? (yes/no): ", bool)
    trade_info['hedging_technology'] = get_validated_input("Do you have sophisticated treasury management systems for hedging activities? (yes/no): ", bool)
    trade_info['has_futures_access'] = get_validated_input("Do you have access to futures markets? (yes/no): ", bool)
    trade_info['has_options_access'] = get_validated_input("Do you have access to options markets? (yes/no): ", bool)
    trade_info['has_international_banking'] = get_validated_input("Do you have established international banking relationships? (yes/no): ", bool)

    print("\nHedging Strategy Preferences:")
    trade_info['transaction_frequency'] = get_validated_input("How frequently does your company engage in foreign currency transactions? (daily/weekly/monthly/quarterly/annually): ", str, valid_options=['daily', 'weekly', 'monthly', 'quarterly', 'annually'])
    trade_info['hedge_ratio'] = get_validated_input("What percentage of your exposure do you want to hedge? (0-100): ", float, valid_range=(0, 100)) / 100
    trade_info['accounting_preference'] = get_validated_input("Do you have a preference for hedge accounting treatment? (yes/no): ", bool)
    trade_info['liquidity_need'] = get_validated_input("Do you need to maintain high liquidity? (yes/no): ", bool)

    print("\nMarket View and Competitive Analysis:")
    trade_info['market_view'] = get_validated_input("What is your view on the direction of this currency pair? (appreciate/depreciate/stable): ", str, valid_options=['appreciate', 'depreciate', 'stable'])
    trade_info['counterparty_risk'] = get_validated_input("On a scale of 0-100, how concerned are you about counterparty risk? (0 = least concerned, 100 = most concerned): ", float, valid_range=(0, 100))
    trade_info['competitive_hedging'] = get_validated_input("Do your main competitors actively hedge their currency exposures? (yes/no/unknown): ", str, valid_options=['yes', 'no', 'unknown'])

    return trade_info

# Calculate score representing the company's resources and capabilities for hedging, which influences the suitability of different hedging methods
def calculate_resource_score(trade_info):
    base_score = 0
    
    # Basic resources
    base_score += sum([
        trade_info['has_treasury_team'],
        trade_info['has_futures_access'],
        trade_info['has_options_access'],
        trade_info['has_international_banking']
    ])
    
    # Company size
    size_multiplier = {'micro': 0.7, 'small': 0.8, 'medium': 0.9, 'large': 1.0}
    base_score *= size_multiplier[trade_info['company_size']]
    
    # Hedging experience
    base_score += trade_info['hedging_experience'] / 20  # 0 to 5 points

    # Hedging technology
    if trade_info['hedging_technology']:
        base_score += 2
    
    # Transaction frequency
    frequency_bonus = {
        'daily': 2, 'weekly': 1.5, 'monthly': 1, 'quarterly': 0.5, 'annually': 0
    }
    base_score += frequency_bonus[trade_info['transaction_frequency']]
    
    # Accounting preference (slight bonus for more sophisticated approach)
    if trade_info['accounting_preference']:
        base_score += 0.5
    
    # Liquidity need (slight penalty as it may limit options)
    if trade_info['liquidity_need']:
        base_score -= 0.5
    
    # Volatility appetite
    base_score += trade_info['volatility_appetite'] / 100  # 0 to 1 point
    
    # Competitive hedging (bonus for market awareness)
    if trade_info['competitive_hedging'] == 'yes':
        base_score += 0.5
    
    # Counterparty risk awareness
    base_score += trade_info['counterparty_risk'] / 200  # 0 to 0.5 points
    
    # Normalize the score to a 0-10 scale
    final_score = min(10, max(0, base_score))
    
    return final_score

# Create and train a personalized model based on user inputs and historical data
def create_and_train_personalized_model(X, y, user_features, model_params):
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    user_features_tensor = torch.FloatTensor(user_features).to(device)
    
    if user_features_tensor.dim() == 3:
        user_features_tensor = user_features_tensor.squeeze(1)
    
    model = EnsembleModel(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        output_size=model_params['output_size'],
        num_user_features=model_params['num_user_features']
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    num_epochs = model_params['num_epochs']
    batch_size = max(2, model_params['batch_size'])  # Ensure minimum batch size of 2
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            batch_user = user_features_tensor[i:i+batch_size]
            
            # Ensure there are at least 2 samples in the batch
            if batch_X.size(0) < 2:
                continue
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_user)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_tensor, user_features_tensor)
            val_loss = criterion(val_outputs, y_tensor)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(X_tensor):.4f}, Val Loss: {val_loss:.4f}')
    
    return model

# Format the currency pair for use with the yfinance API
def format_currency_pair(pair):
    base, quote = pair.split('/')
    return f"{quote}{base}=X"

# Calculate the Relative Strength Index (RSI) technical indicator
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate the Weighted Moving Average (WMA) technical indicator
def weighted_moving_average(data, window):
    weights = np.arange(1, window+1)
    return data.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# Calculate technical indicators for the given market data
def calculate_indicators(data):
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Upper_BB'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_BB'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)

    # Calculate Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = (data['Close'] - low_14) * 100 / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Calculate Average True Range (ATR)
    data['TR'] = np.maximum(data['High'] - data['Low'], 
                            np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                       abs(data['Low'] - data['Close'].shift(1))))
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # Calculate Weighted Moving Averages
    data['WMA50'] = weighted_moving_average(data['Close'], 50)
    data['WMA200'] = weighted_moving_average(data['Close'], 200)

    return data

# Load historical forex data and preprocess it for model training
def load_and_preprocess_data(currency_pair, start_date, end_date):
    formatted_pair = format_currency_pair(currency_pair)
    try:
        data = yf.download(formatted_pair, start=start_date, end=end_date)
        if data.empty:
            print(f"No data available for {currency_pair}. Please check the currency pair and try again.")
            return None, None, None, None

        # Calculate basic features
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        data['RSI'] = calculate_rsi(data['Close'])

        # Calculate advanced indicators
        data = calculate_indicators(data)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Create sequences
        sequence_length = 20  # Use 20 days of data to predict next 5 days
        X, y = [], []
        for i in range(len(data) - sequence_length - 5):
            X.append(data[['Returns', 'Volatility', 'WMA50', 'WMA200', 'RSI', 
                           'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', '%K', '%D', 'ATR']].values[i:i+sequence_length])
            y.append([1 if data['Close'].values[i+sequence_length+5] > data['Close'].values[i+sequence_length] else 0] * len(HEDGING_METHODS))
        
        X = np.array(X)
        y = np.array(y)
        
        # Apply StandardScaler to each feature across all time steps
        scaler = StandardScaler()
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])
        
        return X_scaled, y, scaler, data  # Return the original DataFrame
    except Exception as e:
        print(f"Error fetching data for {currency_pair}: {str(e)}")
        return None, None, None, None

# Use Bayesian optimization to find the best hyperparameters for the model
def optimize_hyperparameters(X, y, user_features):
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train_index, val_index in tscv.split(X):
        global X_train, X_val, y_train, y_val, user_train, user_val
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        user_train, user_val = user_features[train_index], user_features[val_index]
        break
    
    space = [
        Integer(32, 128, name='hidden_size'),
        Integer(1, 2, name='num_layers'),
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
        Integer(30, 80, name='num_epochs'),
        Integer(16, 32, name='batch_size')
    ]

    @use_named_args(space)
    def objective(**params):
        model_params = {
            'input_size': X_train.shape[2],
            'hidden_size': int(params['hidden_size']),
            'num_layers': int(params['num_layers']),
            'output_size': y_train.shape[1],
            'num_user_features': user_train.shape[1],
            'learning_rate': params['learning_rate'],
            'num_epochs': int(params['num_epochs']),
            'batch_size': int(params['batch_size'])
        }
        
        model = create_and_train_personalized_model(X_train, y_train, user_train, model_params)
        
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_val).to(device), torch.FloatTensor(user_val).to(device))
        mse = nn.MSELoss()(y_pred, torch.FloatTensor(y_val).to(device))
        return mse.item()

    res_gp = gp_minimize(objective, space, n_calls=15, random_state=42)
    
    best_params = {
        'hidden_size': int(res_gp.x[0]),
        'num_layers': int(res_gp.x[1]),
        'learning_rate': float(res_gp.x[2]),
        'num_epochs': int(res_gp.x[3]),
        'batch_size': int(res_gp.x[4])
    }
    
    return best_params

# Generate hedging recommendations using the trained neural network model
def neural_network_recommendation(trade_info, model, X, scaler):
    try:
        user_features = torch.FloatTensor([
            trade_info['risk_tolerance'] / 100,
            calculate_resource_score(trade_info) / 10,
            trade_info['duration'] / 365,
            trade_info['hedge_ratio'],
            1 if trade_info['accounting_preference'] else 0,
            1 if trade_info['liquidity_need'] else 0,
            trade_info['volatility_appetite'] / 100,
            {'appreciate': 1, 'depreciate': -1, 'stable': 0}[trade_info['market_view']],
            trade_info['counterparty_risk'] / 100
        ]).unsqueeze(0).to(device)

        latest_data = torch.FloatTensor(X[-1:]).to(device)

        with torch.no_grad():
            predicted_scores = model(latest_data, user_features).squeeze().cpu().numpy()

        recommendations = [
            {"method": method, "score": float(score)}
            for method, score in zip(HEDGING_METHODS, predicted_scores)
        ]

        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return recommendations

    except Exception as e:
        print(f"An error occurred in neural_network_recommendation: {str(e)}")
        return []

# Generate rule-based hedging recommendations based on user inputs and predefined criteria
def recommend_hedging_methods(trade_info):
    recommendations = []
    rts = trade_info["risk_tolerance"]
    rs = calculate_resource_score(trade_info)
    
    def add_recommendation(method, explanation, score):
        recommendations.append({"method": method, "explanation": explanation, "score": score})
    
    # Forward Contracts
    forward_score = 10 + min(trade_info["duration"] / 30, 5) + (rs / 2)
    if trade_info["hedge_ratio"] > 0.7:
        forward_score += 2
    if trade_info["accounting_preference"]:
        forward_score += 1
    if trade_info["volatility_appetite"] < 50:
        forward_score += 2
    add_recommendation(
        "Forward Contracts",
        "Locks in a future exchange rate, providing certainty for your transaction. Suitable for your exposure duration and hedging preferences.",
        forward_score
    )
    
    # Currency Futures
    if trade_info["has_futures_access"]:
        futures_score = 12 + min(trade_info["duration"] / 60, 3) + (rs / 2)
        if trade_info["hedging_experience"] > 50:
            futures_score += 2
        if trade_info["transaction_frequency"] in ["daily", "weekly"]:
            futures_score += 1
        add_recommendation(
            "Currency Futures",
            "Standardized contracts for future currency exchange. Requires more sophistication but can be cost-effective for longer durations.",
            futures_score
        )
    
    # Currency Options
    if trade_info["has_options_access"]:
        options_score = 8 + (rts - 40) / 20 + (rs / 2)
        if trade_info["volatility_appetite"] > 50:
            options_score += 2
        if trade_info["market_view"] != "stable":
            options_score += 1
        if trade_info["liquidity_need"]:
            options_score += 2
        add_recommendation(
            "Currency Options",
            "Provides the right, but not the obligation, to exchange currency at a set rate. Offers flexibility but comes with a premium cost.",
            options_score
        )
    
    # Money Market Hedges
    if trade_info["has_international_banking"]:
        mm_score = 7 + min(trade_info["duration"] / 60, 3) + (rs / 2)
        if trade_info["company_size"] in ["medium", "large"]:
            mm_score += 2
        if trade_info["accounting_preference"]:
            mm_score += 1
        add_recommendation(
            "Money Market Hedges",
            "Involves borrowing in one currency and investing in another. Can be cost-effective but requires good banking relationships.",
            mm_score
        )
    
    # Natural Hedging
    natural_score = 9 + rs
    if trade_info["transaction_frequency"] in ["daily", "weekly"]:
        natural_score += 2
    if trade_info["hedge_ratio"] < 0.5:
        natural_score += 1
    add_recommendation(
        "Natural Hedging",
        "Matching foreign currency inflows and outflows. Cost-effective if applicable to your business operations.",
        natural_score
    )
    
    # Currency Diversification
    if trade_info["amount"] > 100000:
        div_score = 6 + (40 - rts) / 10 + (rs / 2)
        if trade_info["volatility_appetite"] < 50:
            div_score += 2
        if trade_info["competitive_hedging"] == "yes":
            div_score += 1
        add_recommendation(
            "Currency Diversification",
            "Spreading risk across multiple currencies. Reduces exposure to any single currency but requires managing multiple positions.",
            div_score
        )
    
    # Dynamic Hedging
    if rs >= 7:
        dynamic_score = 5 + (rts - 70) / 10 + rs
        if trade_info["hedging_technology"]:
            dynamic_score += 2
        if trade_info["transaction_frequency"] in ["daily", "weekly"]:
            dynamic_score += 2
        add_recommendation(
            "Dynamic Hedging",
            "Continuously adjusting hedge positions based on market conditions. Offers potential for optimization but requires significant resources and expertise.",
            dynamic_score
        )
    
    # Spot Market
    spot_score = 5 + (rts / 20) + (rs / 2)
    if trade_info["duration"] <= 7:
        spot_score += 3
    if trade_info["liquidity_need"]:
        spot_score += 2
    add_recommendation(
        "Spot Market",
        "Immediate exchange at current market rates. Suitable for short-term exposures or when high liquidity is needed.",
        spot_score
    )
    
    # Currency Swaps
    if trade_info["has_international_banking"] and trade_info["duration"] > 90:
        swap_score = 6 + (rs / 2) + min(trade_info["duration"] / 180, 3)
        if trade_info["accounting_preference"]:
            swap_score += 1
        if trade_info["hedge_ratio"] > 0.8:
            swap_score += 2
        add_recommendation(
            "Currency Swaps",
            "Exchange of principal and interest in one currency for principal and interest in another. Suitable for longer-term exposures.",
            swap_score
        )
    
    # Cross-Currency Swaps
    if trade_info["has_international_banking"] and trade_info["duration"] > 180:
        cross_swap_score = 5 + (rs / 2) + min(trade_info["duration"] / 365, 3)
        if trade_info["accounting_preference"]:
            cross_swap_score += 1
        if trade_info["hedge_ratio"] > 0.9:
            cross_swap_score += 2
        add_recommendation(
            "Cross-Currency Swaps",
            "Similar to currency swaps but with exchange of floating interest rates. Suitable for very long-term exposures and interest rate risk management.",
            cross_swap_score
        )
    
    # Proxy Hedging
    proxy_score = 4 + (rs / 2) + (rts / 20)
    if not trade_info["has_futures_access"] and not trade_info["has_options_access"]:
        proxy_score += 2
    if trade_info["hedge_ratio"] < 0.7:
        proxy_score += 1
    add_recommendation(
        "Proxy Hedging",
        "Using a correlated currency or asset to hedge when direct hedging is not possible or cost-effective.",
        proxy_score
    )
    
    # Netting
    if trade_info["transaction_frequency"] in ["daily", "weekly"]:
        netting_score = 7 + (rs / 2)
        if trade_info["company_size"] in ["medium", "large"]:
            netting_score += 2
        if trade_info["hedge_ratio"] < 0.6:
            netting_score += 1
        add_recommendation(
            "Netting",
            "Offsetting receivables and payables to reduce overall exposure. Effective for companies with multiple foreign currency cash flows.",
            netting_score
        )
    
    # Matching
    matching_score = 6 + (rs / 2)
    if trade_info["transaction_frequency"] in ["daily", "weekly", "monthly"]:
        matching_score += 2
    if trade_info["hedge_ratio"] < 0.5:
        matching_score += 1
    add_recommendation(
        "Matching",
        "Aligning foreign currency inflows with outflows. Requires careful cash flow management but can reduce the need for external hedging.",
        matching_score
    )
    
    # Leading and Lagging
    lead_lag_score = 5 + (rs / 2) + (rts / 20)
    if trade_info["market_view"] != "stable":
        lead_lag_score += 2
    if trade_info["liquidity_need"]:
        lead_lag_score += 1
    add_recommendation(
        "Leading and Lagging",
        "Adjusting the timing of payments and receipts based on expected currency movements. Requires market insight and flexible payment terms.",
        lead_lag_score
    )
    
    # Participating Forwards
    if trade_info["has_options_access"]:
        part_forward_score = 7 + (rs / 2) + (rts / 20)
        if trade_info["market_view"] == "appreciate":
            part_forward_score += 2
        if trade_info["hedge_ratio"] > 0.6:
            part_forward_score += 1
        add_recommendation(
            "Participating Forwards",
            "Combination of a forward contract and an option, allowing partial participation in favorable market movements.",
            part_forward_score
        )
    
    # Range Forwards
    if trade_info["has_options_access"]:
        range_forward_score = 6 + (rs / 2) + (rts / 20)
        if 0.4 <= trade_info["hedge_ratio"] <= 0.7:
            range_forward_score += 2
        if trade_info["volatility_appetite"] > 50:
            range_forward_score += 1
        add_recommendation(
            "Range Forwards",
            "Setting a range for the exchange rate, providing protection against adverse movements while allowing some participation in favorable movements.",
            range_forward_score
        )
    
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations


# Adjust recommendation scores based on predicted market conditions to tailor recommendations to expected future market movements
def adjust_scores_based_on_predictions(recommendations, predicted_return, predicted_volatility):
    for rec in recommendations:
        if rec['method'] == "Forward Contracts":
            rec['score'] *= (0.7 - abs(predicted_return) + 0.3 * (1 - predicted_volatility))
        elif rec['method'] == "Currency Futures":
            rec['score'] *= (0.6 - abs(predicted_return) + 0.4 * (1 - predicted_volatility))
        elif rec['method'] == "Currency Options":
            rec['score'] *= (0.5 + predicted_volatility)
        elif rec['method'] == "Money Market Hedges":
            rec['score'] *= (0.6 - abs(predicted_return) + 0.4 * (1 - predicted_volatility))
        elif rec['method'] == "Natural Hedging":
            rec['score'] *= 0.5  # Neutral adjustment
        elif rec['method'] == "Currency Diversification":
            rec['score'] *= (0.4 + predicted_volatility)
        elif rec['method'] == "Participating Forwards":
            rec['score'] *= (0.5 + 0.5 * predicted_return if predicted_return > 0 else 0.5)
        elif rec['method'] == "Range Forwards":
            rec['score'] *= (0.6 - 0.2 * abs(predicted_return) + 0.2 * predicted_volatility)
        elif rec['method'] == "Dynamic Hedging":
            rec['score'] *= (0.3 + predicted_volatility)

    # Renormalize scores
    max_score = max(rec['score'] for rec in recommendations)
    for rec in recommendations:
        rec['score'] /= max_score
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations

# Combine neural network and rule-based recommendations to leverages both data-driven and expert-defined approaches for final recommendations.
def get_combined_recommendations(nn_recommendations, rule_recommendations):
    rule_scores = {rec['method']: rec['score'] for rec in rule_recommendations}
    
    # Find the maximum rule-based score for normalization
    max_rule_score = max(rule_scores.values())
    
    combined_recommendations = []
    for nn_rec in nn_recommendations:
        method = nn_rec['method']
        nn_score = nn_rec['score']
        rule_score = rule_scores.get(method, 0)  # Default to 0 if method not in rule-based recommendations
        
        # Normalize rule-based score to [0, 1] range
        normalized_rule_score = rule_score / max_rule_score
        
        combined_score = 0.6 * nn_score + 0.4 * normalized_rule_score
        
        combined_recommendations.append({
            'method': method,
            'score': combined_score,
            'nn_score': nn_score,
            'rule_score': normalized_rule_score
        })
    
    # Sort recommendations by combined score
    combined_recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return combined_recommendations

# Calculate additional statistical measures for the hedging recommendations
def calculate_additional_stats(trade_info, market_data, volatility):
    amount = trade_info['amount']
    duration = trade_info['duration']
    risk_tolerance = trade_info['risk_tolerance']
    
    # Calculate potential value at risk
    z_score = stats.norm.ppf(1 - risk_tolerance / 100)
    var_daily = amount * volatility * z_score
    var_duration = var_daily * np.sqrt(duration)
    
    # Calculate expected range of exchange rate movement
    current_rate = market_data['Close'].iloc[-1]
    expected_low = current_rate * (1 - volatility * np.sqrt(duration / 365))
    expected_high = current_rate * (1 + volatility * np.sqrt(duration / 365))
    
    # Calculate hedge effectiveness (simplified)
    hedge_effectiveness = min(100, 100 * (1 - (1 - trade_info['hedge_ratio']) * volatility))
    
    # Calculate cost of carry (simplified, assuming interest rate differential of 2%)
    interest_rate_diff = 0.02
    cost_of_carry = amount * interest_rate_diff * (duration / 365)
    
    return {
        "var_daily": var_daily,
        "var_duration": var_duration,
        "expected_low": expected_low,
        "expected_high": expected_high,
        "hedge_effectiveness": hedge_effectiveness,
        "cost_of_carry": cost_of_carry
    }

# Generate a detailed template with steps to implement the recommended hedging strategy
def get_hedging_template(method, trade_info, calculated_values, additional_stats):
    """
    Provide a template with steps to implement the recommended hedging strategy.
    
    :param method: The recommended hedging method
    :param trade_info: Dictionary containing user input
    :param calculated_values: Dictionary containing calculated values (e.g., strike prices, fees)
    :param additional_stats: Dictionary containing additional calculated statistics
    :return: A string with step-by-step instructions
    """
    templates = {
        "Forward Contracts": """
To hedge your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over the next {duration} days, we recommend executing a forward contract. \n Based on your risk tolerance of {risk_tolerance_score} and the forecasted volatility of {volatility:.2f}%, this strategy aims to secure an exchange rate of {forward_rate:.4f}, protecting against potential adverse movements.

Steps to implement Forward Contract hedge:
1. Contact your bank or a forex dealer to set up a forward contract.
2. Specify the following details:
   - Currency pair: {currency_pair}
   - Amount: {amount} {base_currency}
   - Settlement date: {settlement_date}
   - Desired forward rate: {forward_rate}
3. Negotiate the forward rate. Consider the suggested rates:
   - Conservative: {conservative_rate}
   - Neutral: {neutral_rate}
   - Aggressive: {aggressive_rate}
4. Review the contract terms, including any margin requirements.
5. Sign the forward contract agreement.
6. On the settlement date, be prepared to exchange the currencies at the agreed rate.

Additional Information:
- Estimated Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Expected exchange rate range over your exposure period: 
  {expected_low:.4f} to {expected_high:.4f} {currency_pair}
- Estimated cost of carry: {cost_of_carry:.2f} {base_currency}
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Actual rates and fees may vary. Consult with multiple providers for the best terms.
""",

        "Currency Futures": """
To manage your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, we suggest using currency futures for a period of {duration} days. \n Given your access to futures markets and a risk tolerance of {risk_tolerance_score}, locking in a futures contract at a rate of {futures_rate:.4f} can help mitigate the risk associated with exchange rate fluctuations.

Steps to implement Currency Futures hedge:
1. Open an account with a futures broker if you don't have one.
2. Determine the number of contracts needed:
   - Amount to hedge: {amount} {base_currency}
   - Standard contract size: (typically 100,000 or 125,000 of the base currency)
   - Number of contracts: {num_contracts}
3. Choose the appropriate expiration month, closest to but after {settlement_date}.
4. Place an order to {buy_sell} {num_contracts} futures contracts.
5. Monitor the position and be prepared to meet any margin calls.
6. Close out the position before expiration or prepare for delivery.

Additional Information:
- Daily Value at Risk (VaR): {var_daily:.2f} {base_currency}
- Expected exchange rate range over your exposure period: 
  {expected_low:.4f} to {expected_high:.4f} {currency_pair}
- Initial margin requirement (estimated): {initial_margin:.2f} {base_currency}
- Maintenance margin (estimated): {maintenance_margin:.2f} {base_currency}
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Futures trading requires careful management of margin requirements and daily settlement.
""",

        "Currency Options": """
For your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over the next {duration} days, consider purchasing a {option_type} option. \n With an implied volatility of {volatility:.2f}% and a strike price of {strike_price:.4f}, this strategy provides you with the flexibility to benefit from favorable market movements while limiting potential downside risks.

Steps to implement Currency Options hedge:
1. Decide on the type of option:
   - For {trade_type} {base_currency}: Consider a {put_call} option
2. Contact your broker or an options dealer.
3. Specify the following details:
   - Currency pair: {currency_pair}
   - Option type: {put_call}
   - Strike price: Consider {conservative_rate} (conservative) to {aggressive_rate} (aggressive)
   - Expiration date: {expiration_date}
   - Number of contracts: {num_contracts}
4. Get quotes for the option premium at different strike prices.
5. Purchase the option(s) that best fit your risk profile and budget.
6. Monitor the option's value as you approach the expiration date.
7. Exercise the option if favorable, or let it expire if not beneficial.

Additional Information:
- Estimated Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Expected exchange rate range over your exposure period: 
  {expected_low:.4f} to {expected_high:.4f} {currency_pair}
- Estimated option premium (at-the-money): {option_premium:.2f}% of notional value
- Break-even exchange rate: {break_even_rate:.4f} {currency_pair}
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Option premiums can be significant. Compare the cost against potential currency movements.
""",

"Money Market Hedges": """
To hedge your {trade_type} exposure of {amount} {base_currency} in {currency_pair} for the next {duration} days, we recommend a money market hedge. \n By borrowing {base_currency} at an interest rate of {borrowing_rate:.2f}% and investing in {quote_currency} at a rate of {investment_rate:.2f}%, this strategy locks in an effective rate, minimizing currency risk.

Steps to implement Money Market Hedge:
1. Determine the amount in foreign currency: {foreign_amount} {quote_currency}
2. Borrow in your base currency ({base_currency}):
   - Amount to borrow: {borrow_amount} {base_currency}
   - Seek a loan term matching your exposure duration: {duration} days
3. Convert borrowed amount to {quote_currency} at the current spot rate.
4. Invest the {quote_currency} for {duration} days:
   - Seek the best available interest rate for a {duration}-day deposit
5. Use the matured investment to meet your foreign currency obligation.
6. Repay the {base_currency} loan.

Additional Information:
- Estimated Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Expected exchange rate range over your exposure period: 
  {expected_low:.4f} to {expected_high:.4f} {currency_pair}
- Estimated cost of carry: {cost_of_carry:.2f} {base_currency}
- Estimated interest rate differential: {interest_rate_diff:.2f}%
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Compare interest rates carefully. The effectiveness depends on interest rate differentials.
""",

        "Natural Hedging": """
To mitigate your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, consider natural hedging by aligning your revenues and expenses in the same currency over the next {duration} days. \n This method leverages your existing cash flows and reduces reliance on financial instruments, aligning with your moderate risk tolerance of {risk_tolerance_score}.

Steps to implement Natural Hedging:
1. Review your upcoming cash flows in {base_currency} and {quote_currency}:
   - Outflows in {quote_currency}: {amount} {base_currency} equivalent
   - Identify or create offsetting inflows in {quote_currency}
2. Align the timing of your {quote_currency} inflows with the outflow date: {settlement_date}
3. Adjust pricing or payment terms with customers/suppliers to create natural hedges:
   - Consider offering discounts for {quote_currency} payments
   - Negotiate {quote_currency} pricing with suppliers
4. Set up separate {quote_currency} accounts to manage these cash flows.
5. Monitor and match {quote_currency} inflows and outflows regularly.

Additional Information:
- Unhedged Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Expected exchange rate range over your exposure period: 
  {expected_low:.4f} to {expected_high:.4f} {currency_pair}
- Potential savings on transaction costs: {transaction_cost_savings:.2f} {base_currency}
- Recommended buffer for timing mismatches: {timing_buffer:.2f}% of exposure

Note: Natural hedging requires ongoing management and may not provide exact matches.
""",

        "Currency Diversification": """
To hedge your {trade_type} exposure of {amount} {base_currency} in {currency_pair} for the next {duration} days, we recommend diversifying your currency portfolio. \n By allocating your exposure across multiple currencies, this approach spreads the risk and reduces sensitivity to any single currency movement.

Steps to implement Currency Diversification:
1. Analyze your current currency exposures.
2. Identify target currencies for diversification beyond {currency_pair}.
3. For your {amount} {base_currency} exposure:
   - Consider splitting into 3-5 different currencies
   - Example split:
     * 40% in original exposure: {original_exposure} {base_currency}
     * 20% in Currency A: {currency_a_amount} {base_currency} equivalent
     * 20% in Currency B: {currency_b_amount} {base_currency} equivalent
     * 20% in Currency C: {currency_c_amount} {base_currency} equivalent
4. Execute trades to achieve this currency mix.
5. Set up accounts in each currency if necessary.
6. Regularly review and rebalance your currency positions.

Additional Information:
- Estimated reduction in overall currency risk: {risk_reduction:.2f}%
- Expected range of portfolio value fluctuation: 
  {portfolio_low:.2f} to {portfolio_high:.2f} {base_currency}
- Correlation matrix of selected currencies: [Include a small correlation matrix here]
- Recommended rebalancing frequency: {rebalance_frequency}

Estimated fees: Will vary based on the number of currencies and transactions.
Note: Diversification can reduce risk but may also limit potential gains from favorable movements.
""",

        "Dynamic Hedging": """
For managing your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, we recommend a dynamic hedging strategy. \n By adjusting your hedge positions dynamically in response to market movements over the next {duration} days, this approach balances your risk tolerance of {risk_tolerance_score} with the current market volatility of {volatility:.2f}%.

Steps to implement Dynamic Hedging:
1. Set up a system to monitor {currency_pair} exchange rates continuously.
2. Establish rules for adjusting your hedge:
   - Current position: {hedge_ratio}% hedged of {amount} {base_currency}
   - Example rule: Adjust hedge by 5% for every 1% move in exchange rate
3. Define upper and lower limits for your hedge ratio, e.g., 50% - 100%
4. Monitor and forecast {currency_pair} movements:
   - Use technical indicators: RSI, Moving Averages, etc.
   - Consider macroeconomic factors affecting {base_currency} and {quote_currency}
5. Execute trades to adjust your hedge as per your rules:
   - Use a combination of spots, forwards, or options
6. Record each adjustment and its rationale.
7. Regularly review and refine your dynamic hedging rules.

Additional Information:
- Estimated daily hedge adjustment threshold: {daily_threshold:.2f}% move in exchange rate
- Expected number of adjustments over exposure period: {expected_adjustments}
- Potential risk reduction compared to static hedge: {dynamic_risk_reduction:.2f}%
- Estimated additional transaction costs: {additional_transaction_costs:.2f} {base_currency}
- Recommended monitoring frequency: {monitoring_frequency}

Estimated fees: Will vary based on frequency of adjustments and instruments used.
Note: Dynamic hedging requires active management and can incur higher transaction costs.
""",

        "Spot Market": """
For immediate {trade_type} requirements of {amount} {base_currency} in {currency_pair}, consider using the spot market. \n This strategy allows you to execute the transaction at the current market rate of {spot_rate:.4f}, providing a quick solution to meet short-term currency exposure over {duration} days.

Steps to implement Spot Market hedging:
1. Monitor the {currency_pair} exchange rate closely.
2. Decide on your target rate for executing the spot trade.
3. Place a market order or limit order with your forex broker:
   - Currency pair: {currency_pair}
   - Amount: {amount} {base_currency}
   - Type: {buy_sell}
   - Rate: Current spot rate is {spot_rate}
4. Execute the trade when your target rate is reached.
5. Ensure you have sufficient funds in your account for settlement.
6. Be prepared for immediate delivery (usually T+2).

Additional Information:
- Daily exchange rate volatility: {daily_volatility:.2f}%
- Probability of favorable move within {duration} days: {favorable_probability:.2f}%
- Potential saving from optimal timing (estimated): {timing_saving:.2f} {base_currency}
- Recommended limit order range: {limit_low:.4f} to {limit_high:.4f} {currency_pair}
- Maximum adverse move you can tolerate: {max_adverse_move:.2f}%

Estimated fees: {fee_range}
Note: Spot trades offer no protection against adverse market movements before execution.
""",

        "Currency Swaps": """
To hedge your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over the next {duration} days, we recommend a currency swap. \n By exchanging principal and interest payments at a predetermined rate of {swap_rate:.4f}, this strategy can help lock in rates and provide more predictable cash flows.

Steps to implement Currency Swaps hedge:
1. Identify a suitable counterparty for the swap (typically a bank or financial institution).
2. Negotiate the terms of the swap:
   - Notional amount: {amount} {base_currency}
   - Currency pair: {currency_pair}
   - Swap duration: {duration} days
   - Exchange rates for initial and final exchanges
   - Interest rates for each currency
3. Execute the initial exchange of principal.
4. Make periodic interest payments as per the agreed schedule.
5. At maturity, exchange back the principal amounts.

Additional Information:
- Estimated Value at Risk (VaR) for swap duration: {var_duration:.2f} {base_currency}
- Expected interest rate differential benefit: {interest_rate_benefit:.2f} {base_currency}
- Potential counterparty risk exposure: {counterparty_risk:.2f}% of notional value
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: Typically built into the swap rates, consult with swap provider for details.
Note: Currency swaps are complex instruments and may require specific accounting treatment.
""",

        "Cross-Currency Swaps": """
For managing your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, consider a cross-currency swap over {duration} days. \n This approach allows you to exchange interest payments in two different currencies and hedge both currency and interest rate risk simultaneously.

Steps to implement Cross-Currency Swaps hedge:
1. Identify a suitable counterparty for the cross-currency swap.
2. Negotiate the terms of the swap:
   - Notional amounts in both currencies: {amount} {base_currency} and equivalent in {quote_currency}
   - Swap duration: {duration} days
   - Exchange rates for initial and final exchanges
   - Interest rate benchmarks for each currency (e.g., LIBOR, EURIBOR)
3. Execute the initial exchange of principal in both currencies.
4. Make periodic interest payments in respective currencies as per the agreed schedule.
5. At maturity, exchange back the principal amounts in both currencies.

Additional Information:
- Estimated Value at Risk (VaR) for swap duration: {var_duration:.2f} {base_currency}
- Expected interest rate differential benefit: {interest_rate_benefit:.2f} {base_currency}
- Potential counterparty risk exposure: {counterparty_risk:.2f}% of notional value
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: Typically built into the swap rates, consult with swap provider for details.
Note: Cross-currency swaps are complex and may require specific accounting and risk management.
""",

        "Proxy Hedging": """
To hedge your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over the next {duration} days, we recommend a proxy hedge using a correlated currency pair. \n This method leverages market correlation to reduce costs when direct hedging options are less efficient or unavailable.

Steps to implement Proxy Hedging:
1. Identify a suitable proxy currency or asset highly correlated with {currency_pair}.
2. Analyze the historical correlation between the proxy and your target currency pair.
3. Calculate the appropriate hedge ratio based on the correlation:
   - Hedge amount in proxy: {proxy_hedge_amount} {proxy_currency}
4. Execute the hedge using the proxy currency or asset:
   - Use forwards, futures, or options on the proxy
5. Monitor the correlation between the proxy and your target currency pair.
6. Adjust the hedge as necessary if the correlation changes significantly.

Additional Information:
- Correlation coefficient with proxy: {proxy_correlation:.2f}
- Estimated hedge effectiveness using proxy: {proxy_hedge_effectiveness:.2f}%
- Potential basis risk: {basis_risk:.2f}% of hedged amount
- Recommended frequency for correlation review: {correlation_review_frequency}

Estimated fees: Will vary based on the instrument used for proxy hedging.
Note: Proxy hedging may not provide perfect protection and introduces basis risk.
""",

        "Netting": """
To optimize your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, we suggest using a netting strategy. \n By offsetting receivables and payables within your exposure period, this approach minimizes the amount of currency exchanged and reduces transaction costs, enhancing cash flow management."

Steps to implement Netting:
1. Identify all incoming and outgoing cash flows in {currency_pair} over the {duration}-day period.
2. Categorize cash flows by date and counterparty.
3. Calculate net exposure for each date:
   - Net daily exposure: {net_daily_exposure} {base_currency}
4. Implement internal netting procedures:
   - Set up a netting agreement with relevant counterparties if applicable
   - Establish a centralized treasury function to manage netting
5. Hedge the remaining net exposure using other methods as needed.
6. Regularly reconcile and settle netted positions.

Additional Information:
- Estimated reduction in gross exposure: {netting_reduction:.2f}%
- Potential savings on transaction costs: {netting_cost_savings:.2f} {base_currency}
- Recommended netting frequency: {netting_frequency}
- Residual exposure requiring additional hedging: {residual_exposure} {base_currency}

Estimated fees: Minimal for internal netting; may vary for third-party netting services.
Note: Netting can significantly reduce transaction volumes but may require legal agreements.
""",

        "Matching": """
For your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, consider a matching strategy over the next {duration} days. \n This method aligns cash inflows and outflows in the same currency, reducing the need for external hedging instruments and enhancing natural currency management.

Steps to implement Matching:
1. Identify all incoming and outgoing cash flows in {currency_pair} over the {duration}-day period.
2. Align the timing of cash flows where possible:
   - Match {amount} {base_currency} outflow with incoming {quote_currency}
3. Adjust payment terms with customers and suppliers to improve matching:
   - Negotiate earlier/later payment dates to align with your cash flow needs
4. Set up separate accounts for each currency to manage matched flows.
5. For unmatched exposures, consider using other hedging techniques.
6. Regularly review and update your matching strategy.

Additional Information:
- Percentage of exposure matched: {matching_percentage:.2f}%
- Estimated reduction in currency risk: {matching_risk_reduction:.2f}%
- Potential savings on hedging costs: {matching_cost_savings:.2f} {base_currency}
- Recommended review frequency for matching opportunities: {matching_review_frequency}

Estimated fees: Minimal direct fees; consider potential opportunity costs.
Note: Matching can significantly reduce the need for external hedging but requires careful cash flow management.
""",

        "Leading and Lagging": """
To manage your {trade_type} exposure of {amount} {base_currency} in {currency_pair}, we recommend a leading and lagging strategy over {duration} days. \n Adjust payment and collection terms strategically by accelerating or delaying transactions, depending on the expected currency movement and your risk tolerance of {risk_tolerance_score}.

Steps to implement Leading and Lagging:
1. Analyze your {currency_pair} cash flows and market expectations.
2. For {base_currency} payments (outflows):
   - If {quote_currency} is expected to weaken: Consider lagging (delaying) payments
   - If {quote_currency} is expected to strengthen: Consider leading (advancing) payments
3. For {base_currency} receipts (inflows):
   - If {quote_currency} is expected to weaken: Consider leading (accelerating) receipts
   - If {quote_currency} is expected to strengthen: Consider lagging (deferring) receipts
4. Negotiate payment terms with suppliers and customers to implement the strategy.
5. Monitor currency movements and adjust strategy as needed.
6. Be prepared to use other hedging methods if market moves against expectations.

Additional Information:
- Current market view: {market_view} {quote_currency}
- Potential benefit from optimal timing: {leading_lagging_benefit:.2f} {base_currency}
- Maximum suggested lead/lag time: {max_lead_lag_time} days
- Estimated impact on working capital: {working_capital_impact:.2f} {base_currency}

Estimated fees: No direct fees, but consider the time value of money and relationship impacts.
Note: Leading and lagging can be effective but may affect business relationships and working capital.
""",

        "Participating Forwards": """
For your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over {duration} days, consider a participating forward contract. \n This strategy allows you to hedge at a forward rate of {forward_rate:.4f} while maintaining the potential to benefit from favorable market movements up to {participation_rate:.2f}% of the contract value.

Steps to implement Participating Forwards:
1. Contact your bank or a forex dealer offering participating forwards.
2. Specify the details of your exposure:
   - Currency pair: {currency_pair}
   - Amount: {amount} {base_currency}
   - Settlement date: {settlement_date}
3. Negotiate the terms of the participating forward:
   - Protected rate: {protected_rate}
   - Participation percentage: {participation_percentage}%
4. Review and sign the participating forward agreement.
5. Monitor the market and be prepared for settlement:
   - If spot rate is less favorable than protected rate: Exchange at protected rate
   - If spot rate is more favorable: Participate in the upside based on agreed percentage

Additional Information:
- Estimated Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Maximum potential upside participation: {max_upside_participation:.2f} {base_currency}
- Break-even exchange rate: {break_even_rate:.4f} {currency_pair}
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Participating forwards offer downside protection with some upside potential, but at a cost.
""",

"Range Forwards": """
To protect against unfavorable currency movements for your {trade_type} exposure of {amount} {base_currency} in {currency_pair} over {duration} days, we suggest a range forward. \n This approach secures a protective rate range between {lower_bound_rate:.4f} and {upper_bound_rate:.4f}, balancing cost and flexibility given the current market volatility of {volatility:.2f}%.

Steps to implement Range Forwards:
1. Contact your bank or a forex dealer offering range forwards.
2. Specify the details of your exposure:
   - Currency pair: {currency_pair}
   - Amount: {amount} {base_currency}
   - Settlement date: {settlement_date}
3. Negotiate the terms of the range forward:
   - Lower bound rate: {lower_bound_rate}
   - Upper bound rate: {upper_bound_rate}
4. Review and sign the range forward agreement.
5. Monitor the market and be prepared for settlement:
   - If spot rate is below lower bound: Exchange at lower bound rate
   - If spot rate is above upper bound: Exchange at upper bound rate
   - If spot rate is within the range: Exchange at the spot rate

Additional Information:
- Estimated Value at Risk (VaR) for your exposure duration: {var_duration:.2f} {base_currency}
- Maximum potential loss (worst-case scenario): {max_potential_loss:.2f} {base_currency}
- Maximum potential gain (best-case scenario): {max_potential_gain:.2f} {base_currency}
- Hedge effectiveness (based on your hedge ratio): {hedge_effectiveness:.2f}%

Estimated fees: {fee_range}
Note: Range forwards provide flexibility within a specified range but limit both downside protection and upside potential.
"""
    }
    
    template = templates.get(method, "Template not available for this method.")
    
    # Fill in the template with user inputs, calculated values, and additional stats
    try:
        filled_template = template.format(
            currency_pair=trade_info['currency_pair'],
            amount=trade_info['amount'],
            base_currency=trade_info['currency_pair'].split('/')[0],
            quote_currency=trade_info['currency_pair'].split('/')[1],
            settlement_date=(datetime.now() + timedelta(days=trade_info['duration'])).strftime('%Y-%m-%d'),
            duration=trade_info['duration'],
            trade_type='buying' if trade_info['trade_type'] == 'buy' else 'selling',
            buy_sell='Buy' if trade_info['trade_type'] == 'buy' else 'Sell',
            put_call='Put' if trade_info['trade_type'] == 'buy' else 'Call',
            hedge_ratio=trade_info['hedge_ratio'] * 100,
            risk_tolerance_score=trade_info['risk_tolerance'],
            option_premium=calculated_values.get('option_premium', 'N/A'),
            option_type=calculated_values.get('option_type', 'N/A'),
            spot_rate=calculated_values.get('spot_rate', 'N/A'),
            swap_rate=calculated_values.get('swap_rate', 'N/A'),
            forward_rate=calculated_values.get('forward_rate', 'N/A'),
            futures_rate=calculated_values.get('futures_rate', 'N/A'),
            volatility=calculated_values.get('volatility', 'N/A'),
            borrowing_rate=calculated_values.get('borrowing_rate', 'N/A'),
            conservative_rate=calculated_values.get('conservative_rate', 'N/A'),
            initial_margin=calculated_values.get('initial_margin', 'N/A'),
            maintenance_margin=calculated_values.get('maintenance_margin', 'N/A'),
            neutral_rate=calculated_values.get('neutral_rate', 'N/A'),
            aggressive_rate=calculated_values.get('aggressive_rate', 'N/A'),
            fee_range=f"{calculated_values.get('fee_low', 'N/A')} - {calculated_values.get('fee_high', 'N/A')} {trade_info['currency_pair'].split('/')[0]}",
            num_contracts=max(1, round(trade_info['amount'] / 100000)),  # Assuming standard contract size
            foreign_amount=trade_info['amount'] * calculated_values.get('spot_rate', 1),
            borrow_amount=trade_info['amount'] * 1.05,  # Assuming 5% buffer
            original_exposure=trade_info['amount'] * 0.4,
            currency_a_amount=trade_info['amount'] * 0.2,
            currency_b_amount=trade_info['amount'] * 0.2,
            currency_c_amount=trade_info['amount'] * 0.2,
            var_daily=additional_stats['var_daily'],
            var_duration=additional_stats['var_duration'],
            expected_low=additional_stats['expected_low'],
            expected_high=additional_stats['expected_high'],
            hedge_effectiveness=additional_stats['hedge_effectiveness'],
            cost_of_carry=additional_stats['cost_of_carry'],
            expiration_date=calculated_values.get('expiration_date', 'N/A'),
            interest_rate_diff=calculated_values.get('interest_rate_diff', 'N/A'),
            transaction_cost_savings=calculated_values.get('transaction_cost_savings', 'N/A'),
            timing_buffer=calculated_values.get('timing_buffer', 'N/A'),
            risk_reduction=calculated_values.get('risk_reduction', 'N/A'),
            portfolio_low=calculated_values.get('portfolio_low', 'N/A'),
            portfolio_high=calculated_values.get('portfolio_high', 'N/A'),
            rebalance_frequency=calculated_values.get('rebalance_frequency', 'N/A'),
            daily_threshold=calculated_values.get('daily_threshold', 'N/A'),
            expected_adjustments=calculated_values.get('expected_adjustments', 'N/A'),
            dynamic_risk_reduction=calculated_values.get('dynamic_risk_reduction', 'N/A'),
            investment_rate=calculated_values.get('investment_rate', 'N/A'),
            additional_transaction_costs=calculated_values.get('additional_transaction_costs', 'N/A'),
            monitoring_frequency=calculated_values.get('monitoring_frequency', 'N/A'),
            daily_volatility=calculated_values.get('daily_volatility', 'N/A'),
            favorable_probability=calculated_values.get('favorable_probability', 'N/A'),
            timing_saving=calculated_values.get('timing_saving', 'N/A'),
            limit_low=calculated_values.get('limit_low', 'N/A'),
            limit_high=calculated_values.get('limit_high', 'N/A'),
            max_adverse_move=calculated_values.get('max_adverse_move', 'N/A'),
            interest_rate_benefit=calculated_values.get('interest_rate_benefit', 'N/A'),
            counterparty_risk=calculated_values.get('counterparty_risk', 'N/A'),
            proxy_hedge_amount=calculated_values.get('proxy_hedge_amount', 'N/A'),
            proxy_currency=calculated_values.get('proxy_currency', 'N/A'),
            proxy_correlation=calculated_values.get('proxy_correlation', 'N/A'),
            proxy_hedge_effectiveness=calculated_values.get('proxy_hedge_effectiveness', 'N/A'),
            basis_risk=calculated_values.get('basis_risk', 'N/A'),
            correlation_review_frequency=calculated_values.get('correlation_review_frequency', 'N/A'),
            net_daily_exposure=calculated_values.get('net_daily_exposure', 'N/A'),
            netting_reduction=calculated_values.get('netting_reduction', 'N/A'),
            netting_cost_savings=calculated_values.get('netting_cost_savings', 'N/A'),
            netting_frequency=calculated_values.get('netting_frequency', 'N/A'),
            residual_exposure=calculated_values.get('residual_exposure', 'N/A'),
            matching_percentage=calculated_values.get('matching_percentage', 'N/A'),
            matching_risk_reduction=calculated_values.get('matching_risk_reduction', 'N/A'),
            matching_cost_savings=calculated_values.get('matching_cost_savings', 'N/A'),
            matching_review_frequency=calculated_values.get('matching_review_frequency', 'N/A'),
            market_view=trade_info.get('market_view', 'N/A'),
            leading_lagging_benefit=calculated_values.get('leading_lagging_benefit', 'N/A'),
            max_lead_lag_time=calculated_values.get('max_lead_lag_time', 'N/A'),
            working_capital_impact=calculated_values.get('working_capital_impact', 'N/A'),
            protected_rate=calculated_values.get('protected_rate', 'N/A'),
            participation_percentage=calculated_values.get('participation_percentage', 'N/A'),
            max_upside_participation=calculated_values.get('max_upside_participation', 'N/A'),
            break_even_rate=calculated_values.get('break_even_rate', 'N/A'),
            lower_bound_rate=calculated_values.get('lower_bound_rate', 'N/A'),
            upper_bound_rate=calculated_values.get('upper_bound_rate', 'N/A'),
            max_potential_loss=calculated_values.get('max_potential_loss', 'N/A'),
            max_potential_gain=calculated_values.get('max_potential_gain', 'N/A'),
        )
    
    except KeyError as e:
        print(f"Error: Missing key {e} in template. This key is required but not available in the provided data.")
        return f"Error in template formatting: {e}"
    
    return filled_template

# Prepare the calculated values needed for the hedging templates
def prepare_calculated_values(trade_info, market_data, volatility, model):
    # This function should prepare all the necessary calculated values for the hedging templates
    spot_rate = market_data['Close'].iloc[-1]
    forward_points = (trade_info['duration'] / 365) * (market_data['Close'].pct_change().mean() * 365)
    forward_rate = spot_rate * (1 + forward_points)
    
    calculated_values = {
        'spot_rate': spot_rate,
        'forward_rate': forward_rate,
        'futures_rate': forward_rate,
        'strike_price': forward_rate,
        'swap_rate': forward_rate,
        'volatility': volatility,
        'option_type': 'Call' if trade_info['trade_type'] == 'buy' else 'Put',
        'foreign_amount': trade_info['amount'] * spot_rate,
        'borrow_amount': trade_info['amount'] * 1.05,
        'borrowing_rate': 0.03,
        'investment_rate': 0.02,
        'participation_rate': 0.5,
        'conservative_rate': forward_rate * (1 - volatility * np.sqrt(trade_info['duration'] / 365)),
        'neutral_rate': forward_rate,
        'num_contracts': max(1, round(trade_info['amount'] / 100000)),  # Assuming standard contract size of 100,000
        'aggressive_rate': forward_rate * (1 + volatility * np.sqrt(trade_info['duration'] / 365)),
        'fee_low': round(trade_info['amount'] * 0.0005, 2),  # Assuming 0.05% as the lower bound for fees
        'fee_high': round(trade_info['amount'] * 0.002, 2),  # Assuming 0.2% as the upper bound for fees
        'initial_margin': trade_info['amount'] * 0.05,  # Assuming 5% initial margin for futures
        'maintenance_margin': trade_info['amount'] * 0.03,  # Assuming 3% maintenance margin for futures
        'option_premium': volatility * np.sqrt(trade_info['duration'] / 365) * 100,  # Rough estimate
        'break_even_rate': forward_rate * (1 + volatility * np.sqrt(trade_info['duration'] / 365)),
        'interest_rate_diff': 0.02,  # Assuming 2% interest rate differential
        'transaction_cost_savings': trade_info['amount'] * 0.001,  # Assuming 0.1% savings
        'timing_buffer': 5,  # Suggesting a 5% timing buffer
        'risk_reduction': 20,  # Assuming 20% risk reduction through diversification
        'portfolio_low': trade_info['amount'] * 0.95,
        'portfolio_high': trade_info['amount'] * 1.05,
        'rebalance_frequency': 'Monthly',
        'daily_threshold': volatility / 20,  # Daily adjustment threshold as 1/20th of volatility
        'expected_adjustments': trade_info['duration'] / 7,  # Expecting adjustment once a week on average
        'dynamic_risk_reduction': 30,  # Assuming 30% risk reduction through dynamic hedging
        'additional_transaction_costs': trade_info['amount'] * 0.0005 * (trade_info['duration'] / 7),
        'monitoring_frequency': 'Daily',
        'daily_volatility': volatility / np.sqrt(252),
        'favorable_probability': 0.5 + (forward_points / (volatility * np.sqrt(trade_info['duration'] / 365))) / 2,
        'timing_saving': trade_info['amount'] * volatility * 0.1,  # Assuming 10% of volatility as potential saving
        'expiration_date': (datetime.now() + timedelta(days=trade_info['duration'])).strftime('%Y-%m-%d'),
        'limit_low': spot_rate * 0.99,
        'limit_high': spot_rate * 1.01,
        'max_adverse_move': trade_info['risk_tolerance'] / 100 * volatility * np.sqrt(trade_info['duration'] / 365),
        'interest_rate_benefit': trade_info['amount'] * 0.02 * (trade_info['duration'] / 365),
        'counterparty_risk': 1,  # Assuming 1% counterparty risk
        'proxy_hedge_amount': trade_info['amount'] * 0.9,  # Assuming 90% hedge with proxy
        'proxy_currency': 'USD',  # Placeholder, should be determined based on correlation analysis
        'proxy_correlation': 0.8,  # Placeholder, should be calculated based on historical data
        'proxy_hedge_effectiveness': 80,  # Assuming 80% effectiveness of proxy hedge
        'basis_risk': 5,  # Assuming 5% basis risk
        'correlation_review_frequency': 'Weekly',
        'net_daily_exposure': trade_info['amount'] / trade_info['duration'],
        'netting_reduction': 30,  # Assuming 30% reduction in exposure through netting
        'netting_cost_savings': trade_info['amount'] * 0.0005,  # Assuming 0.05% cost savings
        'netting_frequency': 'Daily',
        'residual_exposure': trade_info['amount'] * 0.7,  # Assuming 70% residual exposure after netting
        'matching_percentage': 60,  # Assuming 60% of exposure can be matched
        'matching_risk_reduction': 50,  # Assuming 50% risk reduction through matching
        'matching_cost_savings': trade_info['amount'] * 0.001,  # Assuming 0.1% cost savings
        'matching_review_frequency': 'Weekly',
        'leading_lagging_benefit': trade_info['amount'] * volatility * 0.2,  # Assuming 20% of volatility as potential benefit
        'max_lead_lag_time': min(30, trade_info['duration']),  # Maximum of 30 days or exposure duration
        'working_capital_impact': trade_info['amount'] * 0.1,  # Assuming 10% impact on working capital
        'protected_rate': forward_rate * 0.99,  # Assuming 1% worse than forward rate
        'participation_percentage': 50,  # Assuming 50% participation in favorable movements
        'max_upside_participation': trade_info['amount'] * volatility * 0.5,  # Assuming 50% of volatility as max upside
        'lower_bound_rate': forward_rate * 0.98,  # Assuming 2% below forward rate
        'upper_bound_rate': forward_rate * 1.02,  # Assuming 2% above forward rate
        'max_potential_loss': trade_info['amount'] * 0.02,  # Assuming 2% max loss
        'max_potential_gain': trade_info['amount'] * 0.02,  # Assuming 2% max gain
        'risk_tolerance_score': trade_info['risk_tolerance'],
    }

    return calculated_values

# Main function to orchestrate the entire hedging recommendation process
def main():
    try:
        # Get user input
        trade_info = get_user_input()
        
        # Load and preprocess data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Use 2 years of data
        print(f"Fetching data for {trade_info['currency_pair']} from {start_date} to {end_date}")
        X_scaled, y, scaler, market_data = load_and_preprocess_data(trade_info['currency_pair'], start_date, end_date)
        
        if X_scaled is None or y is None or scaler is None or market_data is None:
            print("Unable to provide recommendations due to data issues.")
            return
        
        print(f"Data shape: X_scaled: {X_scaled.shape}, y: {y.shape}")
        
        # Prepare user features
        user_features = np.array([
            [trade_info['risk_tolerance'] / 100,
             calculate_resource_score(trade_info) / 10,
             trade_info['duration'] / 365,
             trade_info['hedge_ratio'],
             1 if trade_info['accounting_preference'] else 0,
             1 if trade_info['liquidity_need'] else 0,
             trade_info['volatility_appetite'] / 100,
             {'appreciate': 1, 'depreciate': -1, 'stable': 0}[trade_info['market_view']],
             trade_info['counterparty_risk'] / 100]
        ] * len(X_scaled))
        
        print(f"User features shape: {user_features.shape}")
        
        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(X_scaled, y, user_features)
        print("Best hyperparameters:", best_params)
        
        # Prepare model parameters
        model_params = {
            'input_size': X_scaled.shape[2],
            'hidden_size': best_params['hidden_size'],
            'num_layers': best_params['num_layers'],
            'output_size': len(HEDGING_METHODS),
            'num_user_features': user_features.shape[1],
            'learning_rate': best_params['learning_rate'],
            'num_epochs': best_params['num_epochs'],
            'batch_size': best_params['batch_size']
        }
        
        # Train the model with best hyperparameters
        print("Training the model...")
        model = create_and_train_personalized_model(X_scaled, y, user_features, model_params)
        
        # Generate initial recommendations
        print("Generating recommendations...")
        nn_recommendations = neural_network_recommendation(trade_info, model, X_scaled, scaler)
        
        # Predict return and volatility for fine-tuning
        print("Predicting return and volatility...")
        latest_data = X_scaled[-1:]
        latest_user_features = user_features[-1:]
        
        with torch.no_grad():
            predicted_scores = model(torch.FloatTensor(latest_data).to(device), 
                                     torch.FloatTensor(latest_user_features).to(device)).squeeze().cpu().numpy()
        
        # Calculate predicted return and volatility
        current_price = X_scaled[-1, -1, 0]  # Assuming the first feature is Returns
        predicted_return = (predicted_scores[-1] - current_price) / current_price
        predicted_volatility = np.std(predicted_scores) / current_price
        
        print(f"Predicted return: {predicted_return:.4f}")
        print(f"Predicted volatility: {predicted_volatility:.4f}")
        
        # Adjust recommendations based on predictions
        adjusted_recommendations = adjust_scores_based_on_predictions(nn_recommendations, predicted_return, predicted_volatility)
        
        # Generate rule-based recommendations
        rule_recommendations = recommend_hedging_methods(trade_info)
        
        # Combine neural network and rule-based recommendations
        combined_recommendations = get_combined_recommendations(adjusted_recommendations, rule_recommendations)
        
        # Calculate additional statistics
        print("Calculating additional statistics...")
        volatility = market_data['Returns'].std() * np.sqrt(252) if 'Returns' in market_data.columns else 0.15
        additional_stats = calculate_additional_stats(trade_info, market_data, volatility)
        
        # Prepare calculated values
        calculated_values = prepare_calculated_values(trade_info, market_data, volatility, model)
        
        # Display results
        print("\nHedging Recommendations for your Forex Exposure:")
        print(f"Currency Pair: {trade_info['currency_pair']}")
        print(f"Amount: {trade_info['amount']} {trade_info['currency_pair'].split('/')[0]}")
        print(f"Duration: {trade_info['duration']} days")
        print(f"Risk Tolerance Score: {trade_info['risk_tolerance']}")
        print(f"Volatility Appetite: {trade_info['volatility_appetite']}")
        
        print("\nTop 5 Hedging Recommendations:")
        for i, rec in enumerate(combined_recommendations[:5], 1):
            print(f"\n{i}. {rec['method']}:")
            print(f"   Combined suitability score: {rec['score']:.4f}")
            print(f"   Neural Network score: {rec['nn_score']:.4f}")
            print(f"   Rule-based score: {rec['rule_score']:.4f}")
            
            # Get and print the implementation steps
            hedging_template = get_hedging_template(rec['method'], trade_info, calculated_values, additional_stats)
            
            for line in hedging_template.split('\n'):
                print(f"   {line}")
            print("-" * 80)
        
        print("\nRemember: These recommendations, implementation steps, and estimates are based on general market data.")
        print("Actual prices, fees, and terms may vary. Always consult with a qualified financial advisor before making hedging decisions.")
    
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        print("Please try again. If the problem persists, contact support.")

if __name__ == "__main__":
    main()