#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Task - 1 : Predict Restaurant Ratings -: Building a machine learning model to predict the aggregate rating of a restaurant based on other features.
# Step - 1 
#Import Required Libraries
# Data manipulation & analysis
import pandas as pd
import numpy as np

# Visualization libraries for insights
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn: Model building & evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[39]:


# Load the Dataset
# Make sure the file path is correct. Use raw string to avoid errors.
df = pd.read_csv(r'Dataset .csv')  # <- Ensure correct name (watch for space before .csv)
df.head()


# In[40]:


# Step - 2
# Explore the Dataset
# Check data types and null values
df.info()

# View basic statistics (mean, std, etc.)
df.describe()


# In[41]:


# Step - 3
#Handle Missing Values
df = df.dropna(how='any')  
df.isnull().sum()


# In[42]:


# Step 4: Handle Missing Values
# Drop rows with any missing values
df = df.dropna(how='any')

# Check that there are no missing values left
print("Missing values in each column:")
print(df.isnull().sum())


# In[43]:


# Step 5: Encode Categorical Columns
# Identify columns with categorical (object/string) types
cat_cols = df.select_dtypes(include='object').columns

# Apply Label Encoding to convert them into numerical values
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Preview the updated dataset
df.head()


# In[44]:


# Step 6: Clean Column Names (strip spaces, avoid hidden bugs)

df.columns = df.columns.str.strip()  # Removes extra spaces around column names
print("Columns in dataset:", df.columns.tolist())  # Inspect for exact column names


# In[45]:


# Step 7: Define Features and Target

# Update this based on actual column name (from above list)
target_column = 'Aggregate rating'  # <-- Make sure this is correct!

# Separate input features (X) and the target variable (y)
X = df.drop(columns=[target_column])
y = df[target_column]


# In[46]:


# Step 8: Split the Dataset

# Divide data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[47]:


# Step 9: Train Multiple Regression Models and Compare

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = {"MSE": mse, "R² Score": r2}

    # Display performance
    print(f"{name} ➤ MSE: {mse:.3f}, R²: {r2:.3f}")


# In[48]:


# Step 10: Final Model Evaluation (Using Best Performer - Random Forest)

best_model = RandomForestRegressor()
best_model.fit(X_train, y_train)

# Predictions and metrics
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Final Chosen Model: Random Forest")
print(f"MSE: {mse:.3f}")
print(f"R² Score: {r2:.3f}")


# In[49]:


# Step 11: Feature Importance Visualization

# Calculate how much each feature influenced the predictions
importances = pd.Series(best_model.feature_importances_, index=X.columns)

# Sort and plot as horizontal bar chart
importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# In[50]:


# Task 2: Restaurant Recommendation System (Content-Based Filtering) -: Recommending restaurants based on a user's preferences such as cuisine type, price range, and location.
# Step 1: Re-import or use the dataset (clean version from Task 1)
# Dataset `df` is already cleaned and label-encoded from Task 1, so we can reuse it directly.

# Let's take another look at the data
df.head()


# In[51]:


# Step 2: Understand Feature Columns for Recommendation

# Let's recheck the column names to see which can be used for recommendations
print("Available Columns:\n", df.columns.tolist())

# For this content-based system, we will use:
# - 'Cuisines' (user cuisine preference)
# - 'Average Cost for two' (user price range)
# - 'City' or 'Location' (optional for personalization)

# Let's confirm these columns are usable


# In[52]:


# Step 3: Decode Categorical Values Back for Human Readability

# We'll use a separate LabelEncoder to reverse transform encoded columns for recommendation display
# (Assuming 'Cuisines' and 'City' were encoded earlier)

# Load fresh CSV again for decoding and matching user input in raw text form
df_raw = pd.read_csv(r'Dataset .csv')  # This preserves original categorical labels
df_raw.columns = df_raw.columns.str.strip()

# Keep only relevant columns
df_raw = df_raw[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'City', 'Aggregate rating']]
df_raw = df_raw.dropna()
df_raw.reset_index(drop=True, inplace=True)

df_raw.head()


# In[53]:


# Step 4: Build a Content-Based Recommendation Engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a function that prepares the recommendation system
def build_recommendation_engine(df):
    """
    Prepares and returns cosine similarity matrix based on TF-IDF features.
    Uses cuisines and city to determine similarity between restaurants.
    """
    # Combine 'Cuisines' and 'City' into one textual field to use for similarity
    df['combined_features'] = df['Cuisines'] + " " + df['City']

    # Convert combined text into TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Compute cosine similarity between all restaurants
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

# Build similarity matrix
cosine_sim_matrix = build_recommendation_engine(df_raw)


# In[54]:


# Step 5: Create Recommendation Function for Users
def recommend_restaurants(user_cuisine, user_city, price_range, df, sim_matrix, top_n=5,
                          require_booking=False, require_delivery=False):
    """
    Recommend top N restaurants based on cuisine, location, cost, and optional filters.
    Results are ranked by similarity and then sorted by rating.
    """
    df.columns = df.columns.str.strip()  # Ensure no trailing spaces in column names

    # Filter based on user inputs
    filtered_df = df[
        (df['Cuisines'].str.contains(user_cuisine, case=False, na=False)) &
        (df['City'].str.contains(user_city, case=False, na=False)) &
        (df['Average Cost for two'] <= price_range)
    ]

    # Optional filters
    if require_booking and 'Has Table booking' in df.columns:
        filtered_df = filtered_df[df['Has Table booking'] == 'Yes']
    if require_delivery and 'Has Online delivery' in df.columns:
        filtered_df = filtered_df[df['Has Online delivery'] == 'Yes']

    if filtered_df.empty:
        return "⚠️ Sorry, no restaurants match your preferences."

    # Use the first match as the reference point for similarity
    ref_idx = filtered_df.index[0]
    similarity_scores = list(enumerate(sim_matrix[ref_idx]))

    # Sort by similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in similarity_scores[1:top_n*2]]  # more to filter better

    recommended = df.iloc[top_indices]

    # Reapply filters and re-rank by rating
    recommended = recommended[
        (recommended['Cuisines'].str.contains(user_cuisine, case=False, na=False)) &
        (recommended['City'].str.contains(user_city, case=False, na=False)) &
        (recommended['Average Cost for two'] <= price_range)
    ]

    if 'Aggregate rating' in recommended.columns:
        recommended = recommended.sort_values(by='Aggregate rating', ascending=False)

    return recommended[['Restaurant Name', 'Cuisines', 'City', 'Average Cost for two', 'Aggregate rating']].head(top_n).reset_index(drop=True)



# In[55]:


# Step 6: Test the Recommendation System with Sample User Preferences

# Define user preferences
user_cuisine = "Italian"
user_city = "Delhi"
user_price_limit = 800  # In INR

# Get recommendations
recommendations = recommend_restaurants(user_cuisine, user_city, user_price_limit, df_raw, cosine_sim_matrix, top_n=5)

# Show results
print("📌 Top Restaurant Recommendations:")
recommendations


# In[57]:


#  Restaurant Finder =:
# Goal: Find restaurants in Mumbai that serve Italian food,
# cost ₹1000 or less for two people, have table booking, and offer online delivery.
import pandas as pd

# Loading the restaurant data
df = pd.read_csv("Dataset .csv")

# Let's check how many records we have in total
print("Total records:", len(df))

# Step 1: First, pick only those restaurants located in Mumbai
# (we'll make the comparison case-insensitive just to be safe)
step1 = df[df['City'].str.lower() == 'mumbai']
print("After City filter:", len(step1))

# Step 2: From those, we want places that offer Italian cuisine
# Again, keeping it case-insensitive because some entries might use "Italian" or "italian"
step2 = step1[step1['Cuisines'].str.lower().str.contains('italian', na=False)]
print("After Cuisine filter:", len(step2))

# Step 3: Now let's focus on affordable options — ₹1000 or less for two people
step3 = step2[step2['Average Cost for two'] <= 1000]
print("After Price filter:", len(step3))

# Step 4: Filter out only those that offer table booking
# We're checking if the restaurant explicitly says "Yes"
step4 = step3[step3['Has Table booking'].str.lower() == 'yes']
print("After Table Booking filter:", len(step4))

# Step 5: Finally, keep only the restaurants that also allow online delivery
step5 = step4[step4['Has Online delivery'].str.lower() == 'yes']
print("After Online Delivery filter:", len(step5))

# Show the final list — just the top few results to confirm everything worked
print(step5[['Restaurant Name', 'Cuisines', 'City', 'Average Cost for two', 'Aggregate rating']].head())


# In[58]:


# TASK -3 : Cuisine Classification -: Developing a machine learning model to classify restaurants based on their cuisines.
# Importing necessary libraries again if any missed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# In[60]:


# Loading dataset again
df = pd.read_csv("Dataset .csv")


# In[63]:


# STEP 1: DATA CLEANING & PREPROCESSING
# Drop duplicates and fill nulls in Cuisines
df = df.drop_duplicates()
df = df.dropna(subset=['Cuisines'])

# Only keep the primary (first-listed) cuisine as the target class
df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip().lower())

# Optional: Focus only on the top 10 most common cuisines for a balanced classification
top_cuisines = df['Primary Cuisine'].value_counts().nlargest(10).index
df = df[df['Primary Cuisine'].isin(top_cuisines)]

# Encode categorical features
le_city = LabelEncoder()
le_book = LabelEncoder()
le_online = LabelEncoder()

df['City_encoded'] = le_city.fit_transform(df['City'].astype(str))
df['Has Table booking'] = le_book.fit_transform(df['Has Table booking'].astype(str))
df['Has Online delivery'] = le_online.fit_transform(df['Has Online delivery'].astype(str))

# Feature selection: pick numeric and encoded categorical features
features = ['Average Cost for two', 'Price range', 'Votes',
            'City_encoded', 'Has Table booking', 'Has Online delivery']
X = df[features]
y = df['Primary Cuisine']

# Encode target labels
le_cuisine = LabelEncoder()
y_encoded = le_cuisine.fit_transform(y)


# In[64]:


# STEP 2: SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# In[65]:


# STEP 3: MODEL TRAINING
# We'll use a Random Forest for better feature interpretation and non-linearity handling
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# In[66]:


# STEP 4: MODEL EVALUATION
y_pred = model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=le_cuisine.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=le_cuisine.classes_, yticklabels=le_cuisine.classes_)
plt.title("Confusion Matrix - Cuisine Classification")
plt.ylabel("Actual Cuisine")
plt.xlabel("Predicted Cuisine")
plt.tight_layout()
plt.show()

report_df.head(15)  # Show top part of report for main classes


# In[1]:


#Task 4: Location-Based Analysis -: Perform a geographical analysis of the restaurants in the dataset.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster


# In[2]:


# Load the dataset
df = pd.read_csv("Dataset .csv")


# In[3]:


# Step 1: Data Cleaning and Preparation
# Drop missing values related to coordinates and key columns
df = df.dropna(subset=['Latitude', 'Longitude', 'City', 'Aggregate rating'])

# Normalize city names
df['City'] = df['City'].str.title()


# In[5]:


# Step 2: Map Visualization - Plotting restaurant locations on an interactive map
# Create a base map centered on the mean coordinates
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=2, tiles='CartoDB positron')

# Add clustered restaurant markers
marker_cluster = MarkerCluster().add_to(restaurant_map)

# Add data points to the map
for idx, row in df.iterrows():
    popup_info = f"""
    <strong>{row['Restaurant Name']}</strong><br>
    City: {row['City']}<br>
    Rating: {row['Aggregate rating']} ⭐<br>
    Cuisines: {row['Cuisines']}<br>
    Avg Cost: ₹{row['Average Cost for two']}
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_info,
        icon=folium.Icon(color='green' if row['Aggregate rating'] >= 4 else 'blue')
    ).add_to(marker_cluster)

# Save the map as HTML for interactive viewing
restaurant_map.save("Restaurant_Location_Map.html")
restaurant_map.save("Restaurant_Location_Map.html")
restaurant_map.save("Restaurant_Location_Map.html")



# In[6]:


# Step 3: Grouping by City - Restaurant Concentration
city_group = df.groupby('City').agg({
    'Restaurant ID': 'count',
    'Aggregate rating': 'mean',
    'Average Cost for two': 'mean'
}).rename(columns={
    'Restaurant ID': 'Total Restaurants',
    'Aggregate rating': 'Avg Rating',
    'Average Cost for two': 'Avg Cost'
}).sort_values(by='Total Restaurants', ascending=False).head(15)


# In[7]:


# Step 4: Plotting Top Cities by Restaurant Count
plt.figure(figsize=(12, 6))
sns.barplot(data=city_group.reset_index(), x='City', y='Total Restaurants', palette='YlOrBr')
plt.xticks(rotation=45)
plt.title("Top 15 Cities by Number of Restaurants")
plt.ylabel("Number of Restaurants")
plt.xlabel("City")
plt.tight_layout()
plt.show()


# In[8]:


# Step 5: Heatmap of Ratings by City
plt.figure(figsize=(12, 6))
sns.barplot(data=city_group.reset_index(), x='City', y='Avg Rating', palette='Greens')
plt.xticks(rotation=45)
plt.title("Average Restaurant Rating by City")
plt.ylabel("Average Rating")
plt.xlabel("City")
plt.tight_layout()
plt.show()


# In[9]:


# Step 6: Cost Analysis
plt.figure(figsize=(12, 6))
sns.barplot(data=city_group.reset_index(), x='City', y='Avg Cost', palette='Oranges')
plt.xticks(rotation=45)
plt.title("Average Cost for Two by City")
plt.ylabel("₹ Average Cost")
plt.xlabel("City")
plt.tight_layout()
plt.show()

# Output the top insights table
city_group.reset_index()


# In[1]:


def predict_rating_for_input(features):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor

    # Load and clean dataset
    df = pd.read_csv("Dataset .csv")
    df.dropna(subset=['Cuisines'], inplace=True)

    df['City'] = df['City'].astype(str)
    df['Cuisines'] = df['Cuisines'].astype(str)
    df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
    df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

    le_city = LabelEncoder()
    le_cuisine = LabelEncoder()
    df['City_enc'] = le_city.fit_transform(df['City'])
    df['Cuisines_enc'] = le_cuisine.fit_transform(df['Cuisines'])

    df['Price range'] = df['Price range'].astype(int)
    df['Votes'] = df['Votes'].astype(int)

    # Features & Target
    X = df[['Price range', 'Votes', 'City_enc', 'Cuisines_enc', 'Has Table booking', 'Has Online delivery']]
    y = df['Aggregate rating']

    # Train model
    model = RandomForestRegressor()
    model.fit(X, y)

    # User input preprocessing
    user_data = pd.DataFrame([features])
    user_data['City_enc'] = le_city.transform([user_data['city'][0]])
    user_data['Cuisines_enc'] = le_cuisine.transform([user_data['cuisines'][0]])
    user_data['Has Table booking'] = 1 if user_data['has_booking'][0].lower() == "yes" else 0
    user_data['Has Online delivery'] = 1 if user_data['has_delivery'][0].lower() == "yes" else 0

    # RENAME columns to match trained model
    user_data.rename(columns={
        'price_range': 'Price range',
        'votes': 'Votes'
    }, inplace=True)

    # Final model input
    final_input = user_data[['Price range', 'Votes', 'City_enc', 'Cuisines_enc', 'Has Table booking', 'Has Online delivery']]

    prediction = model.predict(final_input)[0]
    return prediction


# In[2]:


def recommend_similar(restaurant_name):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load and clean data
    df = pd.read_csv("Dataset .csv")
    df.dropna(subset=['Cuisines', 'Restaurant Name'], inplace=True)
    df['Restaurant Name Lower'] = df['Restaurant Name'].str.lower()
    restaurant_name = restaurant_name.lower()

    if restaurant_name not in df['Restaurant Name Lower'].values:
        return ["❌ Restaurant not found. Please check the name and try again."]

    # Drop duplicates
    df = df.drop_duplicates(subset='Restaurant Name Lower')
    df.reset_index(drop=True, inplace=True)

    # Create cuisine-based vectors
    vectorizer = CountVectorizer()
    cuisine_matrix = vectorizer.fit_transform(df['Cuisines'])

    # Compute similarity
    similarity = cosine_similarity(cuisine_matrix)

    # Index of the input restaurant
    idx = df[df['Restaurant Name Lower'] == restaurant_name].index[0]

    # Get top 5 similar (excluding itself)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    results = []
    for i in sim_scores:
        name = df.iloc[i[0]]['Restaurant Name']
        cuisines = df.iloc[i[0]]['Cuisines']
        results.append(f"{name} | Cuisines: {cuisines}")

    return results


# In[1]:


def classify_cuisine(data):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    # Load and clean dataset
    df = pd.read_csv("Dataset .csv")
    df.dropna(subset=['Cuisines'], inplace=True)

    # Extract primary cuisine
    df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip().lower())

    # Limit to top 10 cuisines
    top_cuisines = df['Primary Cuisine'].value_counts().nlargest(10).index
    df = df[df['Primary Cuisine'].isin(top_cuisines)]

    # Encode features
    df['City'] = df['City'].astype(str)
    le_city = LabelEncoder()
    df['City_encoded'] = le_city.fit_transform(df['City'])

    df['Has Table booking'] = df['Has Table booking'].map({'Yes': 1, 'No': 0})
    df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': 1, 'No': 0})

    # Encode target
    le_cuisine = LabelEncoder()
    df['Cuisine_encoded'] = le_cuisine.fit_transform(df['Primary Cuisine'])

    # Features & Target
    X = df[['Votes', 'Average Cost for two', 'City_encoded', 'Has Table booking', 'Has Online delivery']]
    y = df['Cuisine_encoded']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prepare input
    user_city = data['city']
    user_votes = data['votes']
    user_cost = data['price']
    booking = 1 if data['booking'].lower() == 'yes' else 0
    delivery = 1 if data['delivery'].lower() == 'yes' else 0

    # Encode input
    if user_city not in le_city.classes_:
        return "⚠️ Unknown city. Please try a common one from the dataset."

    city_encoded = le_city.transform([user_city])[0]

    input_df = pd.DataFrame([{
        'Votes': user_votes,
        'Average Cost for two': user_cost,
        'City_encoded': city_encoded,
        'Has Table booking': booking,
        'Has Online delivery': delivery
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    cuisine = le_cuisine.inverse_transform([prediction])[0]
    return cuisine


# In[3]:


def generate_location_map():
    import pandas as pd
    import folium
    from folium.plugins import MarkerCluster
    import os

    # Load and clean data
    df = pd.read_csv("Dataset .csv")
    df.dropna(subset=['Latitude', 'Longitude', 'Restaurant Name', 'City'], inplace=True)

    # Base map
    center = [df['Latitude'].mean(), df['Longitude'].mean()]
    fmap = folium.Map(location=center, zoom_start=5, tiles='CartoDB positron')
    cluster = MarkerCluster().add_to(fmap)

    for _, row in df.iterrows():
        popup = f"""
        <b>{row['Restaurant Name']}</b><br>
        City: {row['City']}<br>
        Rating: {row['Aggregate rating']}
        """
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup,
            icon=folium.Icon(color='green')
        ).add_to(cluster)

    # Save correctly into static/
    static_dir = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir, exist_ok=True)

    output_path = os.path.join(static_dir, 'Restaurant_Location_Map.html')
    fmap.save(output_path)
    print(f"✅ Map saved at: {output_path}")


# In[ ]:


from ML_Tasks import generate_location_map
generate_location_map()


# In[ ]:




