# SVM Spam Classifier — Streamlit Safe Version

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Spam Email Detector")

emails = [
    "Congratulations! You’ve won a free iPhone",
    "Claim your lottery prize now",
    "Exclusive deal just for you",
    "Act fast! Limited-time offer",
    "Click here to secure your reward",
    "Win cash prizes instantly by signing up",
    "Limited-time discount on luxury watches",
    "Get rich quick with this secret method",
    "Hello, how are you today",
    "Please find the attached report",
    "Thank you for your support",
    "The project deadline is next week",
    "Can we reschedule the meeting to tomorrow",
    "Your invoice for last month is attached",
    "Looking forward to our call later today",
    "Don’t forget the team lunch tomorrow",
    "Meeting agenda has been updated",
    "Here are the notes from yesterday’s discussion",
    "Please confirm your attendance for the workshop",
    "Let’s finalize the budget proposal by Friday"
]

labels = [1]*8 + [0]*12

@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,2),
        max_df=0.9
    )

    X = vectorizer.fit_transform(emails)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = LinearSVC(C=1.0)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return vectorizer, model, accuracy

vectorizer, svm_model, acc = train_model()

st.success(f"Model trained successfully ✅")
st.write(f"Accuracy: **{acc:.2f}**")

user_email = st.text_area("Enter an email message:")

if st.button("Check Spam"):
    if user_email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        email_vec = vectorizer.transform([user_email])
        prediction = svm_model.predict(email_vec)

        if prediction[0] == 1:
            st.error("🚨 This email is SPAM")
        else:
            st.success("✅ This email is NOT spam")

