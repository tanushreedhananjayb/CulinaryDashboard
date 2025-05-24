from flask import Flask, render_template, request, redirect, url_for
import ML_Tasks
import os

app = Flask(__name__)

# Home 
@app.route('/')
def index():
    ML_Tasks.generate_location_map()  # Ensure map is up-to-date when page loads
    return render_template('index.html')

# Task 1: Predict Rating 
@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    features = {
        'price_range': int(request.form['price_range']),
        'votes': int(request.form['votes']),
        'city': request.form['city'],
        'cuisines': request.form['cuisines'],
        'has_booking': request.form['booking'],
        'has_delivery': request.form['delivery']
    }
    prediction = ML_Tasks.predict_rating_for_input(features)
    return render_template('result.html', title="⭐ Predicted Rating", result=f"Estimated Rating: {prediction:.2f} ⭐")

#  Task 2: Recommend Similar Restaurants 
@app.route('/recommend', methods=['POST'])
def recommend():
    restaurant_name = request.form['restaurant']
    recommendations = ML_Tasks.recommend_similar(restaurant_name)
    return render_template('result.html', title="🍜 Recommended Restaurants", result='\n'.join(recommendations))

#  Task 3: Classify Cuisine Type
@app.route('/classify', methods=['POST'])
def classify():
    data = {
        'city': request.form['city'],
        'votes': int(request.form['votes']),
        'price': int(request.form['price']),
        'booking': request.form['booking'],
        'delivery': request.form['delivery']
    }
    cuisine_prediction = ML_Tasks.classify_cuisine(data)
    return render_template('result.html', title="🥗 Predicted Cuisine Type", result=f"Predicted Cuisine: {cuisine_prediction.title()}")

#  Task 4: Embedded Map 
@app.route('/map')
def map_view():
    # (If you use iframe inside index.html, this route may not be needed.)
    return redirect(url_for('static', filename='Restaurant_Location_Map.html'))

#  Main Runner
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use Render's port or default
    app.run(host='0.0.0.0', port=port)

