from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
import os
from joblib import load
import pandas as pd
import tensorflow as tf
import requests
import folium
import geopandas as gpd

# Load the Random Forest model for tabular data
rf_model_tabular = load('random_forest_model3.pkl')

# Initialize Flask application
app = Flask(__name__)

# Preprocessing function for tabular data
def preprocess_tabular_inputs(autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
                               expeditedShipping, seller_in_us, gold, silver, Less_than_week,
                               Less_than_month, more_than_year, high_quality_words,
                               low_quality_words, Calculated, Free, FreePickup, Auction,
                               FixedPrice, bronze, copper, augustus, commodus, domitian,
                               macrinus, nero, nerva, otho, philip, septimius_severus,
                               tiberius, valerian, none_of_above, other_ship):
    return [autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
            expeditedShipping, seller_in_us, gold, silver, Less_than_week,
            Less_than_month, more_than_year, high_quality_words,
            low_quality_words, Calculated, Free, FreePickup, Auction,
            FixedPrice, bronze, copper, augustus, commodus, domitian,
            macrinus, nero, nerva, otho, philip, septimius_severus,
            tiberius, valerian, none_of_above, other_ship]

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to match the input size of the model
    image = cv2.resize(image, (64, 64))
    # Normalize the pixel values
    image = image.astype(float) / 255.0
    # Expand dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/museums')
def museums():
    return render_template('museums.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/predict', methods=['POST'])
def predict_tabular():
    # Get form inputs
    auto_pay = request.form['autoPay']
    top_rated_listing = request.form['topRatedListing']
    best_offer_enabled = request.form['bestOfferEnabled']
    buy_it_now_available = request.form['buyItNowAvailable']
    expedited_shipping = request.form['expeditedShipping']
    seller_in_us = request.form['sellerInUS']
    gold = request.form['gold']
    silver = request.form['silver']
    Less_than_week = request.form['LessThanWeek']
    Less_than_month = request.form['LessThanMonth']
    more_than_year = request.form['MoreThanYear']
    high_quality_words = request.form['highQualityWords']
    low_quality_words = request.form['lowQualityWords']
    Calculated = request.form['Calculated']
    Free = request.form['Free']
    FreePickup = request.form['FreePickup']
    Auction = request.form['Auction']
    FixedPrice = request.form['FixedPrice']
    bronze = request.form['bronze']
    copper = request.form['copper']
    augustus = request.form['augustus']
    commodus = request.form['commodus']
    domitian = request.form['domitian']
    macrinus = request.form['macrinus']
    nero = request.form['nero']
    nerva = request.form['nerva']
    otho = request.form['otho']
    philip = request.form['philip']
    septimius_severus = request.form['septimius_severus']
    tiberius = request.form['tiberius']
    valerian = request.form['valerian']
    none_of_above = request.form['noneOfAbove']
    other_ship = request.form['otherShip']

    # Preprocess form inputs for tabular data
    features = preprocess_tabular_inputs(auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
                                          expedited_shipping, seller_in_us, gold, silver, Less_than_week,
                                          Less_than_month, more_than_year, high_quality_words,
                                          low_quality_words, Calculated, Free, FreePickup, Auction,
                                          FixedPrice, bronze, copper, augustus, commodus, domitian,
                                          macrinus, nero, nerva, otho, philip, septimius_severus,
                                          tiberius, valerian, none_of_above, other_ship)

    # Make prediction using the tabular model
    df = pd.DataFrame([features], columns=['autoPay', 'topRatedListing', 'bestOfferEnabled', 'buyItNowAvailable',
                                           'expeditedShipping', 'seller_in_us', 'gold', 'silver', 'Less_than_week',
                                           'Less_than_month', 'more_than_year', 'high_quality_words',
                                           'low_quality_words', 'Calculated', 'Free', 'FreePickup', 'Auction',
                                           'FixedPrice', 'bronze', 'copper', 'augustus', 'commodus', 'domitian',
                                           'macrinus', 'nero', 'nerva', 'otho', 'philip', 'septimius severus',
                                           'tiberius', 'valerian', 'none_of_above', 'other_ship'])

    prediction_tabular = rf_model_tabular.predict(df)[0]

    return render_template('result.html', prediction_tabular=prediction_tabular)

# Load the TensorFlow model
model_path = r"C:\Users\prach\Downloads\model"
model = tf.keras.models.load_model(model_path)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Check if request contains a file named 'image'
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    # Get the uploaded image file
    image_file = request.files['image']
    
    # Save the image to a temporary location
    temp_image_path = r'C:\Users\prach\OneDrive\Desktop\Mini\mini-6\templates\uploaded_image.png'
    image_file.save(temp_image_path)

    # Preprocess the image
    image = preprocess_image(temp_image_path)

    # Make predictions
    predictions = model.predict(image)

    # Determine the predicted class
    predicted_class = np.argmax(predictions)

    # Determine the final prediction based on the image prediction
    final_prediction = "Roman Republican Coin" if predicted_class == 1 else "Not a Roman Republican Coin"

    # Delete the temporary image file
    os.remove(temp_image_path)

    return render_template('image_result.html', prediction_image=final_prediction)

NUMISTA_API_BASE_URL = "https://api.numista.com/api/v3"

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        search_query = request.form["search_query"]
        search_option = request.form["search_option"]

        url = f"{NUMISTA_API_BASE_URL}/types?q={search_query}&type={'rulers' if search_option == 'ruler' else 'titles'}"
        headers = {"Numista-API-Key": "HzJfHhjVpPNFZiRhpipNSdltQCWkQFYCSwfNVHud"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            results = data.get("rulers" if search_option == "ruler" else "titles", [])
            return render_template("search.html", results=results, search_query=search_query, search_option=search_option)
        else:
            return f"Error: {response.status_code}"
    else:
        return render_template("search.html")
    


@app.route('/mints')
def mints_map():
    # Load GeoJSON data from the provided link
    mints_data = gpd.read_file("https://numismatics.org/chrr/mints.geojson")

    # Create a folium map centered around Italy
    m = folium.Map(location=[42.5, 12.5], zoom_start=6)

    # Define the bounds for Italy
    italy_bounds = [[35, 5], [48, 20]]

    # Set the map bounds to focus on Italy
    m.fit_bounds(italy_bounds)

    # Add mints to the map
    for idx, row in mints_data.iterrows():
        folium.Marker(location=[row.geometry.y, row.geometry.x], popup=f"Mint: {row['name']}\nCount: {row['count']}").add_to(m)

    # Convert the map to HTML
    m_html = m.get_root().render()

    # Pass the HTML content and mint data to the template
    return render_template('mints.html', m_html=m_html, mints=mints_data.to_dict(orient="records"))


@app.route('/findspots')
def findspots_map():
    # Load GeoJSON data for find spots
    findspots_data = gpd.read_file("http://numismatics.org/chrr/findspots.geojson")

    # Create a folium map centered around Europe
    m = folium.Map(location=[42.5, 12.5], zoom_start=4)

    # Define the bounds for Europe
    europe_bounds = [[35, 5], [48, 20]]

    # Set the map bounds to focus on Europe
    m.fit_bounds(europe_bounds)

    # Add find spots to the map
    for idx, row in findspots_data.iterrows():
        folium.Marker(location=[row.geometry.y, row.geometry.x], popup=f"Name: {row['name']}\nCount: {row['count']}").add_to(m)

    # Convert the map to HTML
    n_html = m.get_root().render()

    # Pass the HTML content to the template
    return render_template('findspots.html', n_html=n_html)


if __name__ == '__main__':
    app.run(debug=True)
