import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

class RewardModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, df):
        if 'input' not in df.columns or 'response' not in df.columns or 'rating' not in df.columns:
            raise ValueError("Dataframe must contain 'input', 'response', and 'rating' columns")
        X = self.vectorizer.fit_transform(df["input"] + " " + df["response"])
        y = df["rating"]
        self.model.fit(X, y)

    def predict(self, user_input, response):
        X = self.vectorizer.transform([user_input + " " + response])
        return self.model.predict(X)[0]

    def save(self, path="reward_model.joblib"):
        dump((self.vectorizer, self.model), path)

    def load(self, path="reward_model.joblib"):
        self.vectorizer, self.model = load(path)

if __name__ == "__main__":
    model = RewardModel()
    try:
        df = pd.read_csv("chat_logs.csv")
        df = df[df["rating"] > 0]
        model.train(df)
        model.save()
        print("✅ Reward model retrained and saved as reward_model.joblib")
    except Exception as e:
        print(f"❌ Failed to train reward model: {e}")
