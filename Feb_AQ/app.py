import flask
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import pandas as pd
import pickle
from flask import jsonify
from datetime import datetime, timezone, timedelta
import numpy as np
import shap
import sklearn
from sklearn.preprocessing import MinMaxScaler
import xgboost

import sys

app = Flask(__name__, static_url_path='/static')
print("Flask",flask.__version__)
print("SKlearn", sklearn.__version__)
print("XGboost", xgboost.__version__)
print("panda", pd.__version__)
print("Python version:", sys.version)
# Load the trained model
# Create new scaler and model instances
scaler = MinMaxScaler()

#Diameter
# Step 1: Attempt to load existing models and scalers
try:
    with open("Diameter_xgboost_model_Oct8.pkl", "rb") as model_file:
        model_diameter = pickle.load(model_file)
        print(model_diameter)
except EOFError:
    print("Failed to load diameter model. Please check the file.")

try:
    with open("Diameter_feature_scaler_Oct8.pkl", "rb") as feature_model_file:
        model_feature = pickle.load(feature_model_file)
except EOFError:
    print("Failed to load feature scaler. Creating a new scaler.")
    # Create a new scaler if loading fails

    
try:
    with open("Diameter_target_scaler_Oct8.pkl", "rb") as target_model_file:
        model_target = pickle.load(target_model_file)
except EOFError:
    print("Failed to load feature scaler. Creating a new scaler.")
    # Create a new scaler if loading fails
    
#
print("height")
#Height

with open("Height_xgboost_model_Oct6.pkl", "rb") as height_model_file:
    model_height = pickle.load(height_model_file)



with open("Height_feature_scaler_Oct6.pkl", "rb") as height_feature_model_file:
   height_model_feature = pickle.load(height_feature_model_file)

  
with open("Height_target_scaler_Oct6.pkl", "rb") as height_targer_model_file:
    height_model_target = pickle.load(height_targer_model_file)

#pH

with open("pH_xgboost_model_Oct1.pkl", "rb") as pH_model_file:
    model_pH = pickle.load(pH_model_file)



with open("pH_feature_scaler_Oct1.pkl", "rb") as pH_feature_model_file:
   pH_model_feature = pickle.load(pH_feature_model_file)

  
with open("pH_target_scaler_Oct1.pkl", "rb") as pH_targer_model_file:
    pH_model_target = pickle.load(pH_targer_model_file)


    #TDS

with open("TDS_regression_model_Oct2.pkl", "rb") as TDS_model_file:
    model_TDS = pickle.load(TDS_model_file)



with open("TDS_feature_scaler_Oct2.pkl", "rb") as TDS_feature_model_file:
   TDS_model_feature = pickle.load(TDS_feature_model_file)

  
with open("TDS_target_scaler_Oct2.pkl", "rb") as TDS_targer_model_file:
    TDS_model_target = pickle.load(TDS_targer_model_file)

# Define the feature names used during training XGboost
feature_names_diameter = ['Leaves', 'Height', 'Temp', 'Humidity']
feature_names_height = ['Leaves', 'Diameter', 'Temp', 'Humidity']
feature_names_pH = ['TDS', 'EC', 'Temp']
feature_names_TDS=['pH','Temp','EC']

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/b_predict')
def b_predict():
    return render_template('b_Predict.html')

@app.route('/b_aquaponics')
def b_aquaponics():
    return render_template('b_aquaponics.html')

@app.route('/diameter')
def diameter():
    return render_template('b_diameter.html')

@app.route('/height')
def height():
    return render_template('b_height.html')

@app.route('/ph')
def ph():
    return render_template('b_pH.html')

@app.route('/tds')
def tds():
    return render_template('b_TDS.html')

@app.route("/predict", methods=["POST"])
def predict():
    print("hello")
    if request.method == "POST":
        data = request.json
        leaves = int(data["leaves"])
        height = float(data["height"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
      
        input_data = pd.DataFrame([[leaves, height, temperature, humidity]], columns=feature_names_diameter)
            
            # Perform the prediction using the model
        #diameter = model_diameter.predict(input_data)
        
     # Scale the input data
        scaled_input_data = model_feature.transform(input_data)
        print("Sclaed",scaled_input_data)
        # Perform the prediction using the model
        scaled_input_df = pd.DataFrame(scaled_input_data, columns=feature_names_diameter)
        print("Scaled DataFrame for Prediction:", scaled_input_df)
        print("Prediction input columns:", scaled_input_df.columns)
        print("Going to predict:")
        predicted_scaled = model_diameter.predict(scaled_input_df)
        #print("Prediction",predicted_scaled)
          # Calculate SHAP values for the prediction
        explainer = shap.Explainer(model_diameter)
        shap_values = explainer(scaled_input_df)

        # Extract SHAP values for each feature
        shap_values_list = shap_values.values[0].tolist()
        shap_values_dict = dict(zip(feature_names_diameter, shap_values_list))

        print(shap_values_list)
          # Categorize features based on SHAP values
        

        # Sort the shap_values_dict by SHAP value first, in descending order
        sorted_shap_values = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # Create a list of features in sorted order, without distinguishing positive or negative
        influential_features = ["Temperature" if feature == "Temp" else feature
                               for feature, shap_value in sorted_shap_values]

        # Construct the message
        if influential_features:
            message = f"The factors that influenced the predicted plant diameter the most (in order from highest to lowest) are: {', '.join(influential_features)}."
        else:
            message = "No significant factors influenced the prediction."

        # Inverse transform the prediction
        diameter = model_target.inverse_transform(predicted_scaled.reshape(-1, 1))
        diameter = float(diameter[0])
        diameter = round(diameter, 2)
        
        print("Predicted values (unscaled):", diameter)
        
        # Display the unscaled predictions
        print("Predicted values (unscaled):", np.round(diameter, 2))
        print( message)
 
          
        return jsonify(predicted_diameter=diameter,
                      message1 =message)
       

@app.route("/predict_height", methods=["POST"])
def predict_height():
    print("hello_height")
    if request.method == "POST":
        data = request.json
        leaves = int(data["leaves"])
        diameter = float(data["diameter"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        print("Data captured")
        input_data = pd.DataFrame([[leaves, diameter, temperature, humidity]], columns=feature_names_height)
        print("INput Data",input_data)   
        # Perform the prediction using the model
        #height = model_height.predict(input_data)


        scaled_input_data = height_model_feature.transform(input_data)
        print("Scaled", scaled_input_data)
        # Perform the prediction using the model
        #predicted_scaled = model_height.predict(scaled_input_data)
        scaled_input_df = pd.DataFrame(scaled_input_data, columns=feature_names_height)
        print("Scaled DataFrame for Prediction:", scaled_input_df)
        print("Prediction input columns:", scaled_input_df.columns)
        print("Going to predict:")
        # Perform the prediction using the model
        predicted_scaled = model_height.predict(scaled_input_df)
        print("THe predicted", predicted_scaled)
          # Calculate SHAP values for the prediction
        explainer = shap.Explainer(model_height)
        shap_values = explainer(scaled_input_df)

        # Extract SHAP values for each feature
        shap_values_list = shap_values.values[0].tolist()
        shap_values_dict = dict(zip(feature_names_height, shap_values_list))

        print(shap_values_list)
       
        # Sort the shap_values_dict by SHAP value first, in descending order
        sorted_shap_values = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # Create a list of features in sorted order, without distinguishing positive or negative
        influential_features = [
                                 "Temperature" if feature == "Temp" else feature
                                 for feature, shap_value in sorted_shap_values
                                ]

        # Construct the message
        if influential_features:
            message_height = f"The factors that influenced the predicted plant height the most (in order from highest to lowest) are: {', '.join(influential_features)}."
        else:
            message_height = "No significant factors influenced the prediction."

        # Inverse transform the prediction
        height = height_model_target.inverse_transform(predicted_scaled.reshape(-1, 1))
        height = float(height[0])
        height = round(height, 2)
        
        print("Predicted values (unscaled):", height)
        
        # Display the unscaled predictions
        print("Predicted values (unscaled):", np.round(height, 2))
        print( message_height)
 
          
        return jsonify(predicted_height=height,
                      height_message =message_height)



@app.route("/predict_pH", methods=["POST"])
def predict_pH():
    print("hello_pH")
    if request.method == "POST":
        data = request.json
        tds = int(data["tds"])
        ec = int(data["ec"])
        temperature = float(data["temp"])
       
        print("pHData captured")
        input_data = pd.DataFrame([[tds, ec, temperature]], columns=feature_names_pH)
        print("INput Data",input_data)   
       

        scaled_input_data = pH_model_feature.transform(input_data)
        print("Scaled", scaled_input_data)
        # Perform the prediction using the model
        predicted_scaled = model_pH.predict(scaled_input_data)
        
          # Calculate SHAP values for the prediction
        explainer = shap.Explainer(model_pH)
        shap_values = explainer(scaled_input_data)

        # Extract SHAP values for each feature
        shap_values_list = shap_values.values[0].tolist()
        shap_values_dict = dict(zip(feature_names_pH, shap_values_list))

        print(shap_values_list)
       
        # Sort the shap_values_dict by SHAP value first, in descending order
        sorted_shap_values = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # Create a list of features in sorted order, without distinguishing positive or negative
        influential_features = [
                                 "Temperature" if feature == "Temp" else feature
                                 for feature, shap_value in sorted_shap_values
                                ]

        # Construct the message
        if influential_features:
            message_pH = f"The factors that influenced the predicted water pH the most (in order from highest to lowest) are: {', '.join(influential_features)}."
        else:
            message_pH = "No significant factors influenced the prediction."

        # Inverse transform the prediction
        pH = pH_model_target.inverse_transform(predicted_scaled.reshape(-1, 1))
        pH = float(pH[0])
        pH= round(pH, 2)
        
        print("Predicted values (unscaled):", pH)
        
        # Display the unscaled predictions
        print("Predicted values (unscaled):", np.round(pH, 2))
        print(message_pH)
 
          
        return jsonify(predicted_pH=pH,
                      pH_message =message_pH)

#TDS
@app.route("/predict_TDS", methods=["POST"])
def predict_TDS():
    print("hello_TDS")
    if request.method == "POST":
        data = request.json
        
        pH= float(data["pH"])
        temperature = float(data["temp"])
        ec = int(data["ec"])

       
        print("TDS Data captured")
        input_data = pd.DataFrame([[pH, temperature,ec]], columns=feature_names_TDS)
        print("INput Data",input_data)   
       

        scaled_input_data = TDS_model_feature.transform(input_data)
        print("Scaled", scaled_input_data)
        # Perform the prediction using the model
        predicted_scaled = model_TDS.predict(scaled_input_data)
        
         
        # Inverse transform the prediction
        tds_value = TDS_model_target.inverse_transform(predicted_scaled.reshape(-1, 1))
        
        tds = int(np.round(tds_value[0][0]))
        print("Predicted values (unscaled):", tds)
        
        # Feature Importance Extraction
        if hasattr(model_TDS.named_steps['regressor'], 'coef_'):
            # Get coefficients from the model
            coefficients = model_TDS.named_steps['regressor'].coef_
            
            # Flatten the coefficients if multi-target
            if coefficients.ndim > 1:
                coefficients = coefficients.flatten()
            
            # Get feature names
            feature_names = feature_names_TDS
            
            # Create DataFrame for feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })

            # Add absolute coefficient column for sorting
            feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()

            # Sort by absolute coefficient in descending order
            feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

            # Extract feature names in order of importance
            influential_features = feature_importance['Feature'].tolist()

            # Construct the message based on feature importance
            message_TDS = f"The factors that influenced the predicted water TDS the most (in order from highest to lowest) are: {', '.join(influential_features)}."
        else:
            message_TDS = "The model does not have feature importance available."
        print(message_TDS)
          
        return jsonify(predicted_TDS=tds,
                      TDS_message =message_TDS)



@app.route('/save_option', methods=['POST'])
def save_option():
    # Connect to the SQLite database
    conn = sqlite3.connect('studyDemo1.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Retrieve the selected option from the form
    selected_option = request.form.get('option')
    
    # Insert the selected option into the database
    query = "INSERT INTO selectDHprediction (DHselection) VALUES (?)"
    values = (selected_option,)
    cursor.execute(query, values)
    
    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    
    # Redirect back to the home page or wherever you want after insertion
    return redirect(url_for('home'))



if __name__ == '__main__':
    app.run(debug=True)
