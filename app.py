import streamlit as st
import pandas as pd
import random
import cv2
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from PIL import Image

# Initialize Faker
fake = Faker()

# Define constants
NUM_PARTICIPANTS = 250
DAYS = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
EVENTS = [
    "Solo Singing", "Group Dance", "Photography", "Drama", "Poetry Slam",
    "Stand-up Comedy", "Fashion Show", "Instrumental Music", "Painting", "Debate"
]
COLLEGES = ["ABC University", "XYZ Institute", "LMN College", "PQR Academy", "DEF University"]
STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "West Bengal"]

# Generate dataset
@st.cache_data
def generate_data():
    data = []
    for _ in range(NUM_PARTICIPANTS):
        feedback = fake.sentence(nb_words=12)
        sentiment = TextBlob(feedback).sentiment.polarity  # Sentiment Score

        participant = {
            "Participant_ID": fake.uuid4(),
            "Name": fake.name(),
            "College": random.choice(COLLEGES),
            "State": random.choice(STATES),
            "Day": random.choice(DAYS),
            "Event": random.choice(EVENTS),
            "Age": random.randint(18, 28),
            "Score": round(random.uniform(5, 10), 1),  # Random scores between 5-10
            "Feedback": feedback,
            "Sentiment": sentiment  # Store sentiment score
        }
        data.append(participant)

    df = pd.DataFrame(data)
    return df

# Load dataset
df = generate_data()

# Streamlit UI
# st.set_page_config(page_title="INBLOOM ‘25 Dashboard", layout="wide")

if df.empty:
    st.error("🚨 No data available! Please check the dataset.")
else:
    st.title("🎭 INBLOOM ‘25 - Cultural Events Dashboard")

    # Sidebar Filters
    selected_event = st.sidebar.multiselect("Select Event", df["Event"].unique(), default=df["Event"].unique())
    selected_day = st.sidebar.multiselect("Select Day", df["Day"].unique(), default=df["Day"].unique())

    filtered_df = df[(df["Event"].isin(selected_event)) & (df["Day"].isin(selected_day))]

    st.dataframe(filtered_df)

    # Event-wise Participation Chart
    st.subheader("📊 Event Participation")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=filtered_df, x="Event", order=filtered_df["Event"].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Word Cloud for Feedback
    st.subheader("💬 Word Cloud for Feedback")
    selected_event_wc = st.selectbox("Select Event for Word Cloud", df["Event"].unique())
    feedback_text = " ".join(df[df["Event"] == selected_event_wc]["Feedback"])
    
    if feedback_text:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(feedback_text)
        st.image(wordcloud.to_array(), use_column_width=True)
    else:
        st.warning("⚠ No feedback available for this event.")

    # Sentiment Analysis of Feedback
    st.subheader("📈 Sentiment Analysis of Feedback")
    sentiment_df = df.groupby("Event")["Sentiment"].mean().reset_index()
    sentiment_df = sentiment_df.sort_values(by="Sentiment", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=sentiment_df, x="Event", y="Sentiment", palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Average Sentiment Score")
    plt.title("Event-wise Feedback Sentiment Analysis")
    st.pyplot(fig)

    # Image Processing Module
    st.subheader("📷 Event Image Processing")

    uploaded_file = st.file_uploader("Upload an event-related image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Convert image to numpy array
        img_array = np.array(image)

        # Processing Options
        processing_option = st.selectbox("Choose an Image Processing Option", ["Grayscale", "Edge Detection"])

        if processing_option == "Grayscale":
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")

        elif processing_option == "Edge Detection":
            edges = cv2.Canny(img_array, 100, 200)
            st.image(edges, caption="Edge Detection Image", use_column_width=True, channels="GRAY")

    # Day-wise Image Gallery
    st.subheader("📸 Day-wise Image Gallery")

    day_selected = st.selectbox("Select Day", DAYS)
    
    st.write(f"Showing images for {day_selected}")
    
    sample_images = {
        "Day 1": "https://via.placeholder.com/300x200.png?text=Day+1+Image",
        "Day 2": "https://via.placeholder.com/300x200.png?text=Day+2+Image",
        "Day 3": "https://via.placeholder.com/300x200.png?text=Day+3+Image",
        "Day 4": "https://via.placeholder.com/300x200.png?text=Day+4+Image",
        "Day 5": "https://via.placeholder.com/300x200.png?text=Day+5+Image"
    }

    st.image(sample_images[day_selected], caption=f"Event Image for {day_selected}")

    st.success("✅ Dashboard Loaded Successfully!")
