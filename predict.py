import joblib

# Load the trained model
model = joblib.load("spam_model.pkl")

# Get message from user
message = input("Enter your message: ")

# Predict
prediction = model.predict([message])[0]

# Display result
if prediction == 1:
    print("Result: SPAM")
else:
    print("Result: HAM")