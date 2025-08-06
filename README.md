# Elite Fantasy Football Performance Predictor - Comprehensive Development Prompt

## PROJECT OVERVIEW
Build a production-ready fantasy football performance predictor that leverages advanced statistical modeling, machine learning, and deep learning techniques. This must be an industry/academic-level system with rigorous mathematical foundations and state-of-the-art predictive capabilities.

## CORE REQUIREMENTS

### 1. STATISTICAL & MATHEMATICAL FOUNDATIONS
- **Bayesian Inference**: Implement hierarchical Bayesian models for player performance with prior distributions based on position, team, and historical data
- **Time Series Analysis**: ARIMA, SARIMA, Prophet, and LSTM models for trend analysis and seasonality detection
- **Regression Techniques**: Ridge, Lasso, Elastic Net, Gaussian Process Regression, and Support Vector Regression
- **Monte Carlo Methods**: MCMC sampling for uncertainty quantification and confidence intervals
- **Advanced Statistics**: Implement effect size calculations, confidence intervals, hypothesis testing, and statistical significance testing

### 2. MACHINE LEARNING ARCHITECTURE
- **Ensemble Methods**: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), AdaBoost with proper cross-validation
- **Neural Networks**: Multi-layer perceptrons with dropout, batch normalization, and proper regularization
- **Feature Engineering**: Polynomial features, interaction terms, rolling statistics, exponential moving averages
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for feature space analysis
- **Model Selection**: Automated hyperparameter tuning using Optuna or similar Bayesian optimization

### 3. DEEP LEARNING COMPONENTS
- **LSTM/GRU Networks**: For sequential performance prediction with attention mechanisms
- **Transformer Architecture**: Self-attention for capturing long-range dependencies in player performance
- **Convolutional Networks**: For pattern recognition in performance data
- **Variational Autoencoders**: For learning latent representations of player styles and matchup dynamics
- **Graph Neural Networks**: For modeling team dynamics and player interactions

### 4. REINFORCEMENT LEARNING INTEGRATION
- **Multi-Armed Bandits**: For optimal lineup selection under uncertainty
- **Q-Learning**: For start/sit decision optimization
- **Policy Gradient Methods**: For long-term strategy optimization
- **Thompson Sampling**: For exploration vs exploitation in player selection

### 5. DATA SCIENCE PIPELINE
- **Data Collection**: Historical performance, weather data, injury reports, team statistics, opponent rankings
- **Feature Engineering**: Create advanced metrics like target share volatility, red zone efficiency, pressure rate effects
- **Data Preprocessing**: Proper scaling, encoding, missing value imputation using advanced techniques (KNN, MICE)
- **Outlier Detection**: Isolation Forest, Local Outlier Factor, statistical methods
- **Data Validation**: Schema validation, data drift detection, statistical tests for data quality

### 6. ADVANCED FEATURES TO IMPLEMENT

#### Player Analysis Components:
- **Performance Volatility Modeling**: Calculate and predict week-to-week variance using GARCH models
- **Matchup Analysis**: Opponent defensive rankings, pace of play, weather effects, altitude adjustments
- **Injury Impact Modeling**: Probability distributions for injury effects on performance
- **Usage Rate Predictions**: Target share, snap count, red zone opportunities using regression trees
- **Game Script Analysis**: Predict game flow impact on player usage patterns
- **Ceiling/Floor Calculations**: Statistical bounds based on historical performance distributions

#### Statistical Outputs:
- **Predicted Points**: Mean, median, confidence intervals (80%, 90%, 95%)
- **Start/Sit Recommendation**: Binary classification with probability scores
- **Player Grade**: Normalized score (0-100) based on expected value vs replacement
- **Risk Assessment**: Volatility score and boom/bust probability
- **Matchup Advantage**: Quantified edge based on opponent weaknesses
- **Injury Risk**: Probability of performance decline due to injury status

### 7. TECHNICAL IMPLEMENTATION REQUIREMENTS

#### Backend Architecture:
- **Data Pipeline**: Automated ETL with proper error handling and logging
- **Model Training**: Automated retraining pipeline with A/B testing for model versions
- **API Design**: RESTful API with proper rate limiting and caching
- **Database**: Optimized schema for time-series data with proper indexing
- **Monitoring**: Model performance tracking, drift detection, and alerting

#### Mathematical Rigor:
- **Cross-Validation**: Time-series aware splits, nested CV for hyperparameter tuning
- **Statistical Testing**: Proper train/validation/test splits with statistical significance testing
- **Uncertainty Quantification**: Bayesian neural networks or ensemble methods for prediction intervals
- **Model Interpretability**: SHAP values, LIME explanations for key predictions
- **Performance Metrics**: MAE, RMSE, MAPE, directional accuracy, Sharpe ratio for recommendations

### 8. WEB APPLICATION SPECIFICATIONS

#### Frontend Requirements:
- Clean, professional interface focused on functionality over aesthetics
- Real-time predictions with loading states and error handling
- Interactive charts for historical performance and prediction confidence
- Comparison tools for multiple players
- Export functionality for predictions and analysis

#### Core Pages/Features:
1. **Player Search & Analysis**: Search functionality with autocomplete
2. **Prediction Dashboard**: Main interface showing all key metrics
3. **Historical Performance**: Interactive charts and statistical summaries
4. **Matchup Analyzer**: Head-to-head comparisons and opponent analysis
5. **Batch Processing**: Upload roster for bulk analysis
6. **Model Performance**: Accuracy tracking and model explanations

### 9. QUALITY STANDARDS

#### Code Quality:
- **No Placeholders**: Every function must be fully implemented
- **Error Handling**: Comprehensive try-catch blocks with meaningful error messages
- **Testing**: Unit tests for all statistical functions and model components
- **Documentation**: Inline comments explaining statistical methods and assumptions
- **Optimization**: Efficient algorithms with proper complexity analysis

#### Statistical Validity:
- **Model Validation**: Proper backtesting with walk-forward analysis
- **Significance Testing**: Statistical tests for model improvements
- **Robustness Checks**: Performance across different seasons and conditions
- **Bias Detection**: Analysis for systematic biases in predictions

### 10. DELIVERABLES CHECKLIST

#### Core System:
- [ ] Complete data preprocessing pipeline with advanced feature engineering
- [ ] Multiple trained models (ML, DL, RL) with proper validation
- [ ] Bayesian inference engine for uncertainty quantification
- [ ] Automated model selection and hyperparameter optimization
- [ ] Real-time prediction API with sub-second response times

#### Web Application:
- [ ] Responsive web interface with all specified features
- [ ] Interactive visualizations using advanced charting libraries
- [ ] Real-time data updates and prediction refresh capabilities
- [ ] Export functionality for analysis results
- [ ] Mobile-responsive design

#### Documentation & Validation:
- [ ] Mathematical documentation for all statistical methods
- [ ] Model performance benchmarks and validation results
- [ ] API documentation with usage examples
- [ ] Deployment instructions and system requirements

## TECHNICAL CONSTRAINTS & EXPECTATIONS

1. **No Shortcuts**: Implement full statistical rigor, not simplified approximations
2. **Production Ready**: Code must handle edge cases, errors, and scale properly
3. **Mathematical Accuracy**: All statistical methods must be correctly implemented with proper assumptions
4. **Performance**: Sub-second prediction times for individual players
5. **Maintainability**: Clean, documented code with modular architecture
6. **Extensibility**: System should allow easy addition of new features and models

## SUCCESS CRITERIA

The system will be considered successful when it:
- Demonstrates statistically significant predictive accuracy over baseline models
- Provides reliable uncertainty estimates with proper calibration
- Handles all edge cases without errors or placeholder content
- Delivers professional-grade user experience with comprehensive functionality
- Shows clear mathematical rigor in all statistical computations
- Achieves industry-standard performance benchmarks for similar systems

**CRITICAL**: This is not a prototype or demonstration. Build a complete, production-ready system that meets academic and industry standards for statistical modeling and machine learning applications. Every component must be fully functional with no placeholders or simplified implementations.