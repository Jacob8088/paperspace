import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the CSV file
csv_path = '/home/oqlanguage/data/ard_test.csv'  # Adjust the path to your CSV file
df = pd.read_csv(csv_path)

# Combine columns into a single text column
#df['text'] = df['channel'] + ' ' + df['date'] + ' ' + df['start_time'] + ' ' + df['duration'].astype(str) + ' ' + df['titel']

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# Move the model to the GPU if available
model.to(device)

# Label encoder
label_encoder = LabelEncoder()

# List of categories
categories = [
    'Sports',
    'Entertainment',
    'News',
    'Special Interest',
    'Music',
    'Children',
    'Infotainment',
    'Documentary',
    'Education',
    'Lifestyle',
    'Religious',
    'Adult'
]

# Fit label encoder
label_encoder.fit(categories)

# Function to predict labels
def predict_labels(texts, model, tokenizer, device):
    predicted_labels = []
    progress_bar = tqdm(total=len(texts), desc="Predicting labels", unit="text")
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        predicted_labels.append(predicted_label)
        progress_bar.update(1)  # Update the progress bar after each prediction
    progress_bar.close()  # Close the progress bar after all predictions are done
    return predicted_labels

# Make predictions
#texts = df['text'].tolist()
texts = []
for item in df.to_dict(orient='records'):
    text = (
        f"Given the following details about a TV broadcast, identify its most appropriate category ({categories}): "
        f"Title: {item['title']} Channel: {item['channel']} Air Date: {item['start_time']} "
        "Based on these details, the broadcast should be categorized under the genre of(please just write the category nothing else):"
    )
    texts.append(text)

predicted_labels = predict_labels(texts, model, tokenizer, device)

# Use label encoder to transform predicted labels
predicted_classes = label_encoder.inverse_transform(predicted_labels)

# Add predicted classes to the DataFrame
df['predicted_class'] = predicted_classes

# Save the label encoder
label_encoder_path = "label_encoder.pkl"  # Adjust the path to save the label encoder
joblib.dump(label_encoder, label_encoder_path)

# Save the DataFrame with predictions
output_csv_path = 'ard_predictions.csv'
df[['channel', 'date', 'start_time', 'duration', 'titel', 'predicted_class']].to_csv(output_csv_path, index=False)

print("Predictions saved to", output_csv_path)
