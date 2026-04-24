from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os
import json
import warnings
from dotenv import load_dotenv
import google.generativeai as genai

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# --- LOAD ENV & LLM ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    print("✅ Đã kết nối thành công với Gemini LLM API!")
else:
    print("⚠️ API Key chưa được thiết lập. Hệ thống sẽ sử dụng thuật toán Nhận diện Từ khóa cũ (Hardcode).")

# --- AI MODEL INITIALIZATION ---
print("Đang nạp dữ liệu và huấn luyện AI Model. Vui lòng chờ...")

csv_path = r"e:\EPL\archive (1)\premier_league_complete_stats_until31thGameDayOnSeason2025-26.csv"
df = pd.read_csv(csv_path, encoding='latin-1')

# Lọc cầu thủ và tách đặc trưng
df = df[df['minutesPlayed'] >= 200].reset_index(drop=True)
info_cols = ['player_name', 'team_name', 'position']
features_df = df.drop(columns=info_cols).fillna(0)
COLUMNS_LIST_STR = ", ".join(features_df.columns)

# Chuẩn hoá
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Xây dựng Autoencoder
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

autoencoder.fit(scaled_features, scaled_features, epochs=100, batch_size=32, shuffle=True, verbose=0)
player_embeddings = encoder_model.predict(scaled_features, verbose=0)

print("✅ Huấn luyện Autoencoder thành công! Server sẵn sàng phục vụ.")
# -----------------------------

def tim_cau_thu_bang_cau_hoi(cau_hoi):
    ideal_player = features_df.mean().copy()
    text = cau_hoi.lower()
    vi_tri_yeu_cau = None
    relevant_stats_keys = []
    
    if llm_model:
        # DÙNG AI (LLM) ĐỂ PHÂN TÍCH YÊU CẦU
        prompt = f"""Bạn là chuyên gia phân tích dữ liệu AI Football Scout.
Khách hàng yêu cầu: "{cau_hoi}"
Hãy chọn ra đúng 4 cột dữ liệu bóng đá liên quan NHẤT cần phải đạt mức "TỐI ĐA" để khớp với yêu cầu này. 
Đây là danh sách cột có sẵn trong Datasets: {COLUMNS_LIST_STR}.
Đồng thời dự đoán Position là "D" (Hậu vệ), "M" (Tiền vệ) hoặc "F" (Tiền đạo), hoặc null.

Trả về định dạng JSON bắt buộc sau:
{{
    "position": "F", 
    "stats": [
        {{"col": "tên_cột_1", "label": "Tên tiếng Việt hiển thị (ngắn gọn)"}},
        {{"col": "tên_cột_2", "label": "Tên tiếng Việt hiển thị"}},
        {{"col": "tên_cột_3", "label": "Tên tiếng Việt hiển thị"}},
        {{"col": "tên_cột_4", "label": "Tên tiếng Việt hiển thị"}}
    ]
}}
"""
        try:
            resp = llm_model.generate_content(prompt)
            data = json.loads(resp.text)
            vi_tri_yeu_cau = data.get("position")
            for item in data.get("stats", []):
                col = item.get("col")
                label = item.get("label", col)
                if col in features_df.columns:
                    ideal_player[col] = features_df[col].max()
                    relevant_stats_keys.append((label, col))
        except Exception as e:
            print("LLM Parsing Error:", e)
            
    # FALLBACK: Nếu LLM lỗi hoặc Chưa cài KEY
    if not relevant_stats_keys:
        if any(k in text for k in ["hậu vệ", "phòng ngự", "defender", "cb", "lb", "rb"]):
            vi_tri_yeu_cau = 'D'
        elif any(k in text for k in ["tiền vệ", "midfielder", "cm", "cdm", "cam"]):
            vi_tri_yeu_cau = 'M'
        elif any(k in text for k in ["tiền đạo", "tấn công", "tiền đạo cắm", "striker", "winger", "fw"]):
            vi_tri_yeu_cau = 'F'
            
        if any(k in text for k in ["chuyền", "phát động", "kiến tạo", "lanh lợi", "chia bài", "pass"]):
            ideal_player['accuratePasses'] = features_df['accuratePasses'].max()
            ideal_player['accuratePassesPercentage'] = 100
            ideal_player['totalPasses'] = features_df['totalPasses'].max()
            ideal_player['accurateLongBalls'] = features_df['accurateLongBalls'].max()
            ideal_player['keyPasses'] = features_df['keyPasses'].max()
            ideal_player['assists'] = features_df['assists'].max()
            relevant_stats_keys.extend([("Chuyền bóng", "totalPasses"), ("Tỉ lệ chuyền %", "accuratePassesPercentage"), ("Bóng dài", "accurateLongBalls"), ("Kiến Tạo", "assists"), ("Key Passes", "keyPasses")])
            
        if any(k in text for k in ["tắc bóng", "thu hồi", "giải nguy", "tranh chấp", "chặt chém", "xoạc", "phòng ngự", "phòng thủ"]):
            ideal_player['tacklesWon'] = features_df['tacklesWon'].max()
            ideal_player['clearances'] = features_df['clearances'].max()
            ideal_player['interceptions'] = features_df['interceptions'].max()
            ideal_player['groundDuelsWon'] = features_df['groundDuelsWon'].max()
            relevant_stats_keys.extend([("Tắc bóng", "tacklesWon"), ("Cắt bóng", "interceptions"), ("Giải nguy", "clearances"), ("Thu hồi bóng", "ballRecovery")])
            
        if any(k in text for k in ["ghi bàn", "dứt điểm", "sút", "chớp thời cơ", "bắn phá", "tấn công"]):
            ideal_player['goals'] = features_df['goals'].max()
            ideal_player['shotsOnTarget'] = features_df['shotsOnTarget'].max()
            ideal_player['bigChancesCreated'] = features_df['bigChancesCreated'].max()
            relevant_stats_keys.extend([("Bàn thắng", "goals"), ("Sút trúng đích", "shotsOnTarget"), ("Sút trong vòng cấm", "shotsFromInsideTheBox"), ("Cơ hội tạo ra", "bigChancesCreated")])

        if any(k in text for k in ["tốc độ", "rê bóng", "qua người"]):
            ideal_player['successfulDribbles'] = features_df['successfulDribbles'].max()
            ideal_player['dribbledPast'] = features_df['dribbledPast'].min()
            relevant_stats_keys.extend([("Qua người", "successfulDribbles"), ("Mất bóng", "possessionLost")])
            
        if not relevant_stats_keys:
            ideal_player['rating'] = features_df['rating'].max()
            ideal_player['minutesPlayed'] = features_df['minutesPlayed'].max()
            if vi_tri_yeu_cau == 'F' or vi_tri_yeu_cau == 'M' or vi_tri_yeu_cau is None:
                ideal_player['goals'] = features_df['goals'].max()
                ideal_player['assists'] = features_df['assists'].max()
                ideal_player['shotsOnTarget'] = features_df['shotsOnTarget'].max()
            elif vi_tri_yeu_cau == 'D':
                ideal_player['tacklesWon'] = features_df['tacklesWon'].max()
                ideal_player['interceptions'] = features_df['interceptions'].max()
            relevant_stats_keys = [("Số phút", "minutesPlayed"), ("Rating", "rating"), ("Bàn thắng", "goals"), ("Kiến tạo", "assists")]
    
    # Lấy 4 chỉ số nổi bật nhất
    display_stats = []
    seen = set()
    for label, col in relevant_stats_keys:
        if col not in seen:
            seen.add(col)
            display_stats.append((label, col))
            if len(display_stats) >= 4:
                break
        
    ideal_scaled = scaler.transform([ideal_player.values])
    ideal_embedding = encoder_model.predict(ideal_scaled, verbose=0)
    
    if vi_tri_yeu_cau is not None and vi_tri_yeu_cau in ['D', 'M', 'F']:
        idx_list = df[df['position'] == vi_tri_yeu_cau].index
    else:
        idx_list = df.index
        
    if len(idx_list) == 0:
        return []

    player_embeddings_subset = player_embeddings[idx_list]
    sim_scores = cosine_similarity(ideal_embedding, player_embeddings_subset)[0]
    
    top_indices_local = sim_scores.argsort()[-5:][::-1]
    
    results = []
    for rank, idx_local in enumerate(top_indices_local, 1):
        real_idx = idx_list[idx_local]
        match_score = float(sim_scores[idx_local]) * 100
        stats_list = []
        for label, col in display_stats:
            val = float(df.loc[real_idx, col])
            stats_list.append({"label": label, "value": f"{val:.1f}" if val % 1 != 0 else str(int(val))})
            
        player_dict = {
            "name": str(df.loc[real_idx, 'player_name']),
            "team": str(df.loc[real_idx, 'team_name']),
            "position": str(df.loc[real_idx, 'position']),
            "match": f"{match_score:.1f}%",
            "dynamic_stats": stats_list
        }
        results.append(player_dict)
        
    return results

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"reply": "Xin lỗi, hình như bạn chưa nhập thông tin!"})
    
    players = tim_cau_thu_bang_cau_hoi(message)
    
    reply_text = f"Dựa vào phân tích thông minh của hệ thống đối với yêu cầu: '{message}', tôi xuất ra được mảng thông số kỹ năng ẩn đáng giá nhất và đây là đề xuất cho bạn:"
    if 'cảm ơn' in message.lower() or 'chào' in message.lower():
        reply_text = "Chào vị HLV trưởng! Tôi là **AI Football Scout**. Hãy nói cho tôi biết ông cần tăng cường sức mạnh gì cho đội bóng vào mùa giải tới?"
        players = []

    return jsonify({
        "reply": reply_text,
        "players": players
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
