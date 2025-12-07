import streamlit as st
import pandas as pd
import datetime
import sys
import os

# Yolu ekle
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from inference import preprocess_and_predict, get_category_average
except ImportError as e:
    st.error(f"HATA: inference.py yüklenemedi. {e}")
    st.stop()

st.set_page_config(page_title="App Score AI", layout="centered", page_icon="📱")

# Header
st.title(" 📱 Google Play | App Success Predictor")
st.markdown("Uygulamanızın özelliklerini girin, yapay zeka **başarı puanını** ve **iyileştirme önerilerini** sunsun.")
st.divider()

# --- FORM ---
with st.form("main_form"):
    c1, c2 = st.columns(2)
    
    with c1:
        app_name = st.text_input("Uygulama Adı", "Candy Crush Saga")
        category = st.selectbox("Kategori", [
            'GAME', 'FAMILY', 'TOOLS', 'BUSINESS', 'MEDICAL', 
            'PRODUCTIVITY', 'PERSONALIZATION', 'COMMUNICATION', 
            'SPORTS', 'LIFESTYLE', 'FINANCE', 'EDUCATION', 
            'PHOTOGRAPHY', 'SHOPPING'
        ])
        reviews = st.number_input("Yorum Sayısı", 0, value=500)
        installs = st.number_input("İndirme Sayısı", 0, value=10000)
        
    with c2:
        app_type = st.selectbox("Tür", ['Free', 'Paid'])
        price = st.number_input("Fiyat ($)", 0.0, value=0.0)
        content_rating = st.selectbox("Hedef Kitle", ['Everyone', 'Teen', 'Mature 17+', 'Everyone 10+'])
        last_updated = st.date_input("Son Güncelleme", datetime.date(2018, 8, 1))
        size = st.text_input("Boyut (Örn: 15M)", "15M")

    submit = st.form_submit_button("✨ Analiz Et", type="primary")

# --- SONUÇ EKRANI ---
if submit:
    # Veri Hazırlığı
    input_data = {
        'App': app_name,
        'Category': category,
        'Reviews': reviews,
        'Size': size,
        'Installs': installs,
        'Type': app_type,
        'Price': price,
        'Content Rating': content_rating,
        'Genres': category,
        'Last Updated': last_updated.strftime('%Y-%m-%d'),
        'Current Ver': '1.0',
        'Android Ver': '4.0'
    }
    
    try:
        # 1. Hesaplamalar
        score = preprocess_and_predict(input_data)
        avg_score = get_category_average(category)
        diff = score - avg_score
        
        st.divider()
        st.subheader("📊 Analiz Sonuçları")
        
        # 2. Metrikler (KPI)
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            st.metric("Tahmini Puan", f"{score:.2f} / 5.0", delta=f"{diff:.2f} Sektör Ort. Farkı")
        
        with kpi2:
            st.metric("Sektör Ortalaması", f"{avg_score:.2f}", delta_color="off")
            
        with kpi3:
            # Renkli Durum
            if score >= 4.5:
                st.success("🌟 Süper Star!")
            elif score >= 4.0:
                st.info("✅ Başarılı")
            elif score >= 3.5:
                st.warning("⚠️ Ortalama")
            else:
                st.error("🛑 Kritik")

        # 3. Görsel Bar
        st.write("Başarı Skalası:")
        progress_color = "red" if score < 3.5 else "orange" if score < 4.2 else "green"
        st.progress(score / 5.0)
        
        st.divider()
        
        # 4. Yapay Zeka Tavsiyeleri (Actionable Insights)
        st.subheader("🤖 Yapay Zeka Tavsiyeleri")
        
        # Güncellik Analizi
        days_diff = (datetime.date.today() - last_updated).days
        if days_diff > 90:
            st.warning(f"📅 **Güncellik Uyarısı:** Uygulamanız {days_diff} gündür güncellenmemiş. Güncel tutmak puanı artırır.")
        else:
            st.success("📅 **Güncellik:** Harika! Uygulamanız güncel.")
            
        # Etkileşim Analizi (Interaction Rate)
        # 0'a bölme hatasını önlemek için +1
        int_rate = reviews / (installs + 1)
        if int_rate < 0.01:
            st.warning(f"💬 **Etkileşim Düşük ({int_rate:.1%}):** İndirenler yorum yapmıyor. Uygulama içine 'Bizi Puanlayın' butonu ekleyin.")
        elif int_rate > 0.05:
            st.success(f"💬 **Etkileşim Yüksek ({int_rate:.1%}):** Kullanıcılar uygulamanızı konuşuyor, bu çok iyi!")
            
        # Başlık Analizi
        if len(app_name) > 60:
            st.error("📝 **İsim Çok Uzun:** Spam olarak algılanabilir. Daha kısa ve akılda kalıcı bir isim seçin.")
            
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")