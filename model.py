# ============================================================
#  Fake News Detection — Model Training
#  Final Year Project
# ============================================================

import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── Setup ──────────────────────────────────────────────────
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())


# ─── Step 1: Load Data ──────────────────────────────────────
# Dataset: ISOT Fake News Dataset (Fake.csv and True.csv)
# Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df['label'] = 0   # 0 = Fake
real_df['label'] = 1   # 1 = Real

print(f"Fake articles : {len(fake_df)}")
print(f"Real articles : {len(real_df)}")


# ─── Step 2: Combine & Shuffle ──────────────────────────────
df = pd.concat([fake_df, real_df], ignore_index=True)

# Use title + body text together for better accuracy
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[['content', 'label']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total samples : {len(df)}")


# ─── Step 3: Clean Text ─────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)       # keep only letters
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

df['clean'] = df['content'].apply(clean_text)
print("Text cleaning done.")


# ─── Step 4: Split Data ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']   # keep class balance in both splits
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ─── Step 5: TF-IDF Vectorization ───────────────────────────
# Converts text into numeric features the model can learn from.
# ngram_range=(1,2) means it captures single words AND two-word phrases.

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"Vocabulary size : {len(vectorizer.vocabulary_)}")


# ─── Step 6: Train Model ────────────────────────────────────
# Logistic Regression works very well for text classification.
# It's fast, interpretable, and performs ~98% on this dataset.

model = LogisticRegression(max_iter=1000, C=5, random_state=42)
model.fit(X_train_vec, y_train)
print("Model training done.")


# ─── Step 7: Evaluate ───────────────────────────────────────
predictions = model.predict(X_test_vec)
accuracy    = accuracy_score(y_test, predictions)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Fake', 'Real']))


# ─── Step 8: Save Model & Vectorizer ────────────────────────
joblib.dump(model,      "classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Saved: classifier.pkl")
print("Saved: vectorizer.pkl")
