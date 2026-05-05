# House_Price_Prediction
🏠 Housing Price Prediction System
📖 Overview
This project is an end-to-end Machine Learning web application that predicts housing prices based on property features. It demonstrates the complete ML workflow — from data preprocessing and model building to deployment using a Flask web interface.
The application allows users to input property details and receive real-time price predictions.
________________________________________
🎯 Objectives
•	Build a reliable regression model for house price prediction
•	Apply data preprocessing and feature engineering techniques
•	Compare multiple models and select the best-performing one
•	Deploy the model using a web-based interface
________________________________________
⚙️ Tech Stack
•	Language: Python
•	Libraries: NumPy, Pandas, Scikit-learn
•	Backend Framework: Flask
•	Frontend: HTML, CSS
•	Model Storage: Pickle
________________________________________
🧠 Machine Learning Approach
Data Processing
•	Categorical encoding (City, Country)
•	Feature selection and transformation
•	Train-test split (80/20)
Models Implemented
•	Linear Regression
•	Ridge Regression
•	Lasso Regression
Evaluation Metrics
•	R² Score
•	Root Mean Squared Error (RMSE)
Ridge Regression is selected as the final model based on performance.
________________________________________
🚀 Application Workflow
1.	User inputs property details via the web interface
2.	Input data is converted into numerical format
3.	Data is scaled using StandardScaler
4.	Trained model predicts the house price
5.	Prediction is displayed on the UI
________________________________________
📊 Input Features
•	Bedrooms
•	Bathrooms
•	Living Area (sqft)
•	Lot Area (sqft)
•	Floors
•	Waterfront
•	View Rating
•	Condition
•	Above Ground Area
•	Basement Area
•	Year Built
•	Year Renovated
•	City (Encoded)
•	Country (Encoded)
________________________________________
⚠️ Limitations
•	City input is encoded numerically (not user-friendly)
•	Model performance depends on dataset quality
•	Basic UI with limited validation
________________________________________
🔮 Future Enhancements
•	Replace numeric city input with dropdown selection
•	Implement advanced models (Random Forest, XGBoost)
•	Add input validation and error handling
•	Deploy on cloud platforms (AWS / Render)
•	Improve UI/UX design
________________________________________
🤝 Contribution
Contributions are welcome. Please open an issue or submit a pull request for improvements.
________________________________________
💬 Support
For queries or collaboration:
•	📧 Email: prajwalwankhede011.com
•	📱 Phone: 8888317087
If you found this project useful, consider giving it a ⭐ on GitHub.
________________________________________


