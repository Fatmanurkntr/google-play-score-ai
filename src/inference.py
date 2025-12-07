import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
import os

# Uyarıları kapat
warnings.filterwarnings('ignore')

# Dosya yollarını dinamik olarak belirle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.join(CURRENT_DIR, '..')
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed_google_play_data.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'best_model.pkl')

# Global Değişkenler
try:
    TRAIN_DATA = pd.read_csv(DATA_PATH)
    MODEL = joblib.load(MODEL_PATH)
    
    # Content Rating için Encoder'ı hazırla
    le_content = LabelEncoder()
    le_content.fit(TRAIN_DATA['Content Rating'].astype(str))
    
    # Kategori Ortalamalarını Önceden Hesapla (Hız için)
    CATEGORY_MEANS = TRAIN_DATA.groupby('Category')['Rating'].mean().to_dict()
    GLOBAL_MEAN = TRAIN_DATA['Rating'].mean()
    
except Exception as e:
    raise FileNotFoundError(f"Kritik dosyalar yüklenemedi: {e}")

# Modelin Beklediği 15 Özellik
FINAL_FEATURES = [
    'Reviews', 'Size', 'Installs', 'Price', 'Title_Length', 'Is_Pro_App', 'Days_Since_Update', 
    'Reviews_Log', 'Installs_Log', 'Category_Encoded', 'Interaction_Rate', 'Primary_Genre_Encoded', 
    'App_Cluster', 'Type_Encoded', 'Content_Rating_Encoded'
]

def get_category_average(category_name: str) -> float:
    """Seçilen kategorinin ortalama puanını döndürür."""
    return CATEGORY_MEANS.get(category_name, GLOBAL_MEAN)

def preprocess_and_predict(input_data: dict) -> float:
    """Kullanıcı verisini işler ve tahmin üretir."""
    
    df = pd.DataFrame([input_data])
    
    # --- TEMEL TEMİZLİK ---
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce').fillna(0)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
    
    # Size Dönüşümü (Esnek Giriş: 15, 15M, 15MB)
    def clean_size_input(val):
        s = str(val).upper().replace('M', '').replace('K', '').replace('B', '').replace('+', '').strip()
        try:
            # Eğer kullanıcı '15M' girdiyse 15000000, sadece '15' girdiyse 15000000 varsayalım (MB olarak)
            num = float(s)
            if num < 1000: # Muhtemelen MB girildi
                return num * 1000000
            return num
        except:
            return TRAIN_DATA['Size'].median()
            
    df['Size'] = df['Size'].apply(clean_size_input)

    # --- FEATURE ENGINEERING ---
    df['Reviews_Log'] = np.log1p(df['Reviews'])
    df['Installs_Log'] = np.log1p(df['Installs'])
    df['Title_Length'] = df['App'].apply(len)
    df['Is_Pro_App'] = df['App'].apply(lambda x: 1 if any(k in str(x).lower() for k in ['pro', 'premium', 'paid']) else 0)
    
    # Tarih
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    max_date = pd.to_datetime(TRAIN_DATA['Last Updated']).max()
    df['Days_Since_Update'] = (max_date - df['Last Updated']).dt.days.fillna(0)
    
    # Interaction Rate
    df['Interaction_Rate'] = df['Reviews'] / (df['Installs'] + 1)
    
    # Genre
    df['Primary_Genre'] = df['Genres'].str.split(';').str[0]
    
    # --- ENCODING ---
    genre_map = TRAIN_DATA.groupby('Primary_Genre')['Rating'].mean()
    df['Primary_Genre_Encoded'] = df['Primary_Genre'].map(genre_map).fillna(GLOBAL_MEAN)
    
    cat_map = TRAIN_DATA.groupby('Category')['Rating'].mean()
    df['Category_Encoded'] = df['Category'].map(cat_map).fillna(GLOBAL_MEAN)
    
    df['Type_Encoded'] = df['Type'].map({'Free': 0, 'Paid': 1}).fillna(0)
    
    try:
        df['Content_Rating_Encoded'] = le_content.transform(df['Content Rating'].astype(str))
    except:
        df['Content_Rating_Encoded'] = 0
        
    df['App_Cluster'] = 0 

    # --- TAHMİN ---
    prediction = MODEL.predict(df[FINAL_FEATURES])
    return float(np.clip(prediction[0], 1.0, 5.0))