import pandas as pd
import pickle
import base64
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
import cv2
import os
import string
from itertools import groupby
import glob

df = pd.read_csv('test.csv')
df['text'] = df['text'].fillna('')

targets = ["like", "comment", "hide", "expand", "open_photo", "open", "share_to_message"]
def preprocess_text(text):
    text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])
    return text.lower()

df['processed_text'] = df['text'].apply(preprocess_text)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

tfidf_matrix = vectorizer.transform(df['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)
df['count_chars'] = df['text'].apply(lambda x: len(str(x)))

def calculate_regex_text_features_new(text):
    if pd.isna(text):
        text = ""
    features = {}
    features['count_numeric_sequences'] = len(re.findall(r"\d+", text))
    features['count_alphanumeric_words'] = len(re.findall(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b", text))
    features['count_emails'] = len(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text))
    features['count_urls'] = len(re.findall(r"http[s]?://(?:[a-zA-Z0-9$-_@.&+!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+", text))
    features['count_words_starting_with_vowel'] = len(re.findall(r"\b[AEIOUaeiou][a-zA-Z]*\b", text))
    features['count_words_ending_with_consonant'] = len(re.findall(r"\b[a-zA-Z]*[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\b", text))
    features['count_repeated_chars'] = len(re.findall(r"(.)\1\1+", text))
    features['count_consecutive_upper'] = len(re.findall(r"[A-Z]{2,}", text))
    features['count_parentheses'] = len(re.findall(r"[()]", text))
    features['count_brackets'] = len(re.findall(r"[\[\]]", text))
    features['count_braces'] = len(re.findall(r"[{}]", text))
    features['count_underscores'] = len(re.findall(r"_", text))
    features['count_percent_signs'] = len(re.findall(r"%", text))
    features['count_dollar_signs'] = len(re.findall(r"\$", text))
    features['count_ampersands'] = len(re.findall(r"&", text))
    features['count_hashes'] = len(re.findall(r"#", text))
    features['count_at_symbols'] = len(re.findall(r"@", text))
    features['count_ellipses'] = len(re.findall(r"\.{3,}", text))
    features['count_hyphenated_words'] = len(re.findall(r"\b\w+-\w+\b", text))
    features['count_words_with_apostrophes'] = len(re.findall(r"\b\w+'\w+\b", text))
    features['count_sentences'] = len(re.findall(r"[.!?]+", text))
    features['count_decimal_numbers'] = len(re.findall(r"\b\d+\.\d+\b", text))
    features['count_camelcase_words'] = len(re.findall(r"\b(?=\w*[a-z])(?=\w*[A-Z])\w+\b", text))
    features['count_numeric_word_tokens'] = len(re.findall(r"\b\d+\b", text))
    features['count_alphabetic_tokens'] = len(re.findall(r"\b[a-zA-Z]+\b", text))
    features['count_non_alphabetic_tokens'] = len(re.findall(r"\b[^a-zA-Z\s]+\b", text))
    features['count_words_with_multiple_vowels'] = len(re.findall(r"\b(?=(?:.*[aeiouAEIOU]){3,})\w+\b", text))
    features['count_words_with_repeated_letters'] = len(re.findall(r"\b\w*(\w)\1\w*\b", text))
    features['count_contractions'] = len(re.findall(r"\b\w+'(s|re|ve|ll|d|t)\b", text))
    features['count_emoticons'] = len(re.findall(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", text))
    return features

df_features = df['text'].apply(calculate_regex_text_features_new)
df_features = pd.DataFrame(df_features.tolist())
df = pd.concat([df.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

def calculate_symmetry(gray):
    h, w = gray.shape
    if w % 2 != 0:
        gray = gray[:, :-1]
    left_half = gray[:, :w//2]
    right_half_flipped = np.fliplr(gray[:, w//2:])
    return -np.mean(np.abs(left_half - right_half_flipped))

def extract_features(img):
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        means_rgb = rgb.mean(axis=(0,1))
        stds_rgb = rgb.std(axis=(0,1))
        means_hsv = hsv.mean(axis=(0,1))
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        noise = np.abs(gray - blur).mean()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        symmetry = calculate_symmetry(gray)
        white_pixels = np.sum(gray > 250) / gray.size
        black_pixels = np.sum(gray < 5) / gray.size
        texture = gray.std()
        mean_r, mean_g, mean_b = means_rgb
        color_temp = mean_r - mean_b
        return np.concatenate([
            [w, h],
            means_rgb,
            stds_rgb,
            means_hsv,
            [sharpness, noise, symmetry, 
             white_pixels, black_pixels, 
             texture, color_temp]
        ])
    except Exception as e:
        return np.zeros(17)

def batch_process(images, batch_size=1000):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_image, images), total=len(images)))
    return np.vstack(results)

def process_image(base64_str):
    try:
        img = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return extract_features(img)
    except:
        return np.zeros(17)
feature_cols = [
    'width', 'height',
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_h', 'mean_s', 'mean_v',
    'sharpness', 'noise', 'symmetry',
    'white_pixels', 'black_pixels',
    'texture', 'color_temp'
]

features = batch_process(df['photo'])
df = pd.concat([df, pd.DataFrame(features, columns=feature_cols)], axis=1)

def count_features(text):
    if pd.isna(text):
        return [0] * 15
    vowels = set('aeiouAEIOU')
    punctuation = set(string.punctuation)
    words = text.split()
    count_periods = text.count('.')
    count_commas = text.count(',')
    count_exclamations = text.count('!')
    count_questions = text.count('?')
    count_digits = sum(c.isdigit() for c in text)
    count_words = len(words)
    count_vowels = sum(c in vowels for c in text)
    count_consonants = sum(c.isalpha() and c not in vowels for c in text)
    count_whitespace = sum(c.isspace() for c in text)
    count_lowercase = sum(c.islower() for c in text)
    count_uppercase = sum(c.isupper() for c in text)
    count_punctuation = sum(c in punctuation for c in text)
    count_special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
    count_newlines = text.count('\n')
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    return [
        count_periods, count_commas, count_exclamations, count_questions, count_digits,
        count_words, count_vowels, count_consonants, count_whitespace, count_lowercase,
        count_uppercase, count_punctuation, count_special_chars, count_newlines, avg_word_length
    ]

feature_cols = [
    'count_periods', 'count_commas', 'count_exclamations', 'count_questions', 'count_digits',
    'count_words', 'count_vowels', 'count_consonants', 'count_whitespace', 'count_lowercase',
    'count_uppercase', 'count_punctuation', 'count_special_chars', 'count_newlines', 'avg_word_length'
]

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

target_cols = ["like", "comment", "hide", "expand", "open_photo", "open", "share_to_message"]

model_tf = TFAutoModelForSequenceClassification.from_pretrained("rubert_finetuned_multi_target")
tokenizer = AutoTokenizer.from_pretrained("rubert_finetuned_multi_target")
model_tf.config.output_hidden_states = True

texts = df['text'].tolist()
batch_size = 32
num_samples = len(texts)
all_logits = []
all_embeddings = []
for i in range(0, num_samples, batch_size):
    batch_texts = texts[i:i+batch_size]
    encoded_inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="tf"
    )
    outputs = model_tf(encoded_inputs, training=False)
    batch_logits = outputs.logits.numpy()
    all_logits.append(batch_logits)
    batch_embeddings = outputs.hidden_states[-1][:, 0, :].numpy()
    all_embeddings.append(batch_embeddings)
logits_full = np.concatenate(all_logits, axis=0)
embeddings_full = np.concatenate(all_embeddings, axis=0)
df_results = pd.DataFrame(texts, columns=["text"])
for i, col in enumerate(target_cols):
    df[f'pred_{col}'] = logits_full[:, i]
embedding_dim = embeddings_full.shape[1]
for j in range(embedding_dim):
    df[f"emb_{j}"] = embeddings_full[:, j]

from catboost import CatBoostRegressor
df[feature_cols] = df['text'].apply(lambda text: pd.Series(count_features(text)))
col_to_drop = targets + ['photo', 'text', 'processed_text']
def load_models_and_predict(X_inference):
    X = X_inference.drop(col_to_drop, axis=1, errors='ignore')
    model_files = glob.glob("*.cbm")
    predictions = {target: [] for target in targets}
    for model_path in model_files:
        target_name = os.path.basename(model_path).split("_fold")[0]
        model = CatBoostRegressor()
        model.load_model(model_path)
        pred = model.predict(X)
        predictions[target_name].append(pred)
    final_predictions = {}
    for target in targets:
        avg_pred = np.mean(predictions[target], axis=0)
        final_predictions[target] = avg_pred * df['view'].values
    return pd.DataFrame(final_predictions)
predictions_df = load_models_and_predict(df)
for i in predictions_df.columns:
    for j in range(len(predictions_df)):
        predictions_df[i][j] = max(predictions_df[i][j], 0)
predictions_df.to_csv('submission.csv', index=False)
