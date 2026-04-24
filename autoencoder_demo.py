import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings

warnings.filterwarnings('ignore')

# 1. Load du lieu
csv_path = 'archive (1)/premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv'
df = pd.read_csv(csv_path, encoding='latin-1')

# Loc
df = df[df['minutesPlayed'] >= 200].reset_index(drop=True)

# Chuan bi feature
info_cols = ['player_name', 'team_name', 'position']
features_df = df.drop(columns=info_cols).fillna(0)

# Chuan hoa
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Autoencoder
input_dim = scaled_features.shape[1]
encoding_dim = 16

tf.random.set_seed(42)

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
bottleneck = Dense(encoding_dim, activation='relu', name="embedding_layer")(encoded)
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.fit(
    scaled_features, scaled_features,
    epochs=100,
    batch_size=32,
    shuffle=True,
    verbose=0 
)

player_embeddings = encoder_model.predict(scaled_features, verbose=0)
similarity_matrix = cosine_similarity(player_embeddings)

def recommend_players(player_name, top_n=5):
    if player_name not in df['player_name'].values:
        print(f"Khong tim thay cau thu: {player_name}")
        return
    idx = df[df['player_name'] == player_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_scores = [i[1] for i in sim_scores[1:top_n+1]]
    print(f"==========================================")
    print(f"Ban dang tim kiem cau thu thay the cho: {player_name}")
    print(f"==========================================")
    for index, score in zip(top_indices, top_scores):
        p_name = df.loc[index, 'player_name']
        team = df.loc[index, 'team_name']
        pos = df.loc[index, 'position']
        print(f" -> {p_name} ({team} | {pos}) | Do tuong dong: {score*100:.2f}%")

print("Hoan thanh huan luyen Autoencoder. Dang goi y ket qua...")
recommend_players('Declan Rice', top_n=5)
recommend_players('Bukayo Saka', top_n=5)
recommend_players('Micky van de Ven', top_n=5)

