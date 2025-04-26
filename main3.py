import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import shap
import gym
from gym import spaces
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from stable_baselines3 import DQN

# Load Dataset
df = pd.read_csv(r"C:\Users\CANOSSA\Documents\Projects\Credit Fraud\Credit fraud detection.csv")

# Preprocessing
df = df.drop(columns=["nameOrig", "nameDest"])
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])
X = df.drop(columns=["isFraud", "isFlaggedFraud"])
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle Class Imbalance
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Feature Importance
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_}).sort_values(by="Importance", ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance in Fraud Detection")
plt.show()

# Graph-Based Features
G = nx.from_pandas_edgelist(df, source="nameOrig", target="nameDest", edge_attr=["amount", "type"], create_using=nx.DiGraph())
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)
df["degree_centrality"] = df["nameOrig"].apply(lambda x: degree_centrality.get(x, 0))
df["pagerank"] = df["nameOrig"].apply(lambda x: pagerank.get(x, 0))

# Explainability with SHAP
explainer = shap.Explainer(xgb_model, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Reinforcement Learning Environment
class FraudEnv(gym.Env):
    def __init__(self):
        super(FraudEnv, self).__init__()
        self.state = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(X_train.shape[1],), dtype=np.float32)

    def reset(self):
        self.idx = np.random.randint(0, len(X_train))
        self.state = X_train[self.idx]
        return self.state

    def step(self, action):
        reward = 1 if action == y_train.iloc[self.idx] else -1
        done = True  # Ensure episode ends after one step
        return self.state, reward, done, {}

# Train RL Model
rl_model = DQN("MlpPolicy", FraudEnv(), verbose=1)
rl_model.learn(total_timesteps=10000)

# Model Evaluation
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    print(f"📊 {name} Model Performance:")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print("-" * 40)

evaluate_model(rf_model, "Random Forest")
evaluate_model(xgb_model, "XGBoost")

def evaluate_rl(env, model, X_test):
    rewards = []
    for i in range(len(X_test)):
        obs = X_test[i]
        action, _ = model.predict(obs)
        rewards.append(1 if action == y_test.iloc[i] else 0)
    print(f"📊 RL Model Accuracy: {np.mean(rewards):.4f}")

evaluate_rl(FraudEnv(), rl_model, X_test)
