import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Function to clean text (basic preprocessing)
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return " ".join(text.lower().split())  # Lowercasing & basic cleaning

# Function to get sentiment score
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to classify sentiment with GIFs
def classify_sentiment(score):
    if score > 0.5:
        return "Positive", "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif"  # Happy GIF
    elif score < -0.5:
        return "Negative", "https://media.giphy.com/media/3og0INyCmHlNylks9O/giphy.gif"  # Angry GIF
    else:
        return "Neutral", "https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif"  # Neutral GIF

# Streamlit App UI Enhancements
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="wide")
st.markdown(
   """
    <style>
        /* Apply gradient background to the main app */
        .stApp { 
            background: linear-gradient(135deg, #2874F0, #FFD700); 
            background-attachment: fixed;
            background-size: cover;
        }

        /* Title Styling */
        .title { 
            text-align: center; 
            color:  white; 
            font-size: 32px; 
            font-weight: bold; 
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        }

        /* Subtitle Styling */
        .subtitle { 
            text-align: center; 
            color: #fff400; 
            font-size: 18px; 
            font-weight: bold;
        }

        /* Sidebar Background Gradient */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #FFD700, #2874F0); 
            color: white;
        }

        /* Upload Section */
        .upload-section {
            background-color: #0097ff ; /* Semi-transparent yellow */
            padding: 20px; 
            border-radius: 10px; 
            text-align: center; 
            font-weight: bold;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        }

        /* Sentiment Box */
        .sentiment-box {
            padding: 15px; 
            border-radius: 8px; 
            font-size: 20px; 
            font-weight: bold; 
            text-align: center;
            color: white;
            background: linear-gradient(135deg, #2874F0, #FFD700);
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='title'>📝 Sentiment Analysis App</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze customer sentiments with AI-powered insights!</p>", unsafe_allow_html=True)

# File upload section with styled box
st.sidebar.header("📂 Upload a CSV File")
st.sidebar.markdown("<div class='upload-section'>📌 Upload a CSV file containing reviews.</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"], help="Upload a CSV file containing reviews.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check if 'Review_Text' column exists
    if 'Review_Text' in df.columns:
        st.sidebar.success("✅ File uploaded successfully!")
        
        # Preprocess and analyze sentiment
        df['Cleaned_Review'] = df['Review_Text'].apply(preprocess_text)
        df['Sentiment_Score'] = df['Cleaned_Review'].apply(get_sentiment)
        df[['Sentiment_Label', 'GIF_URL']] = df['Sentiment_Score'].apply(lambda x: pd.Series(classify_sentiment(x)))
        
        # Display results
        st.subheader("📊 Sentiment Analysis Results")
        st.dataframe(df[['Review_Text', 'Sentiment_Label']])
        
        # Plot sentiment distribution
        st.subheader("📈 Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df['Sentiment_Label'], palette=["yellow", "red", "green"], ax=ax)
        st.pyplot(fig)
        
        # Download button for results
        st.sidebar.download_button("📥 Download Results", df.to_csv(index=False), file_name="sentiment_results.csv", mime="text/csv")
    else:
        st.sidebar.error("❌ The uploaded file must contain a 'Review_Text' column.")

# Text Input for User Review Prediction
st.subheader("📝 Test Sentiment on Custom Review")
user_input = st.text_area("Enter a review:", placeholder="Type your review here...")
if user_input:
    sentiment_score = get_sentiment(user_input)
    sentiment_label, gif_url = classify_sentiment(sentiment_score)

    # Define color based on sentiment
    sentiment_color = "#28a745" if "Positive" in sentiment_label else "#dc3545" if "Negative" in sentiment_label else "#ffc107"

    # Display result with GIF
    st.markdown(
        f"<div class='sentiment-box' style='background-color:{sentiment_color}; color:white;'>"
        f"🎯 <b>Predicted Sentiment:</b> {sentiment_label} (Score: {sentiment_score:.2f})</div>",
        unsafe_allow_html=True
    )
    st.image(gif_url, width=250)  # Display GIF below sentiment result
