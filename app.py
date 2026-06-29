import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# KONFIGURASI HALAMAN & TEMA DI STREAMLIT
# ==========================================
st.set_page_config(
    page_title="SRQ-20 Screening & AI Analytics",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: white !important;
        border: none !important;
    }
    .status-badge {
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        font-size: 1.1rem;
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
    }
    .badge-stress {
        background-color: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid #F43F5E;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA PERTANYAAN RESMI SRQ-20 (WHO)
# ==========================================
srq_questions = [
    {"id": 1, "text": "Apakah selama 30 hari terakhir ini Anda sering menderita sakit kepala?", "cat": "Fisik / Somatik"},
    {"id": 2, "text": "Apakah selama 30 hari terakhir ini Anda tidak nafsu makan?", "cat": "Fisik / Somatik"},
    {"id": 3, "text": "Apakah selama 30 hari terakhir ini Anda sulit tidur?", "cat": "Fisik / Somatik"},
    {"id": 4, "text": "Apakah selama 30 hari terakhir ini Anda mudah takut?", "cat": "Kecemasan / Emosional"},
    {"id": 5, "text": "Apakah selama 30 hari terakhir Anda merasa tegang, cemas, atau kuatir?", "cat": "Kecemasan / Emosional"},
    {"id": 6, "text": "Apakah selama 30 hari terakhir ini tangan Anda gemetar?", "cat": "Fisik / Somatik"},
    {"id": 7, "text": "Apakah selama 30 hari terakhir ini pencernaan Anda terganggu / buruk?", "cat": "Fisik / Somatik"},
    {"id": 8, "text": "Apakah selama 30 hari terakhir ini Anda sulit berpikir jernih?", "cat": "Kognitif"},
    {"id": 9, "text": "Apakah selama 30 hari terakhir ini Anda merasa tidak bahagia?", "cat": "Depresi / Emosional"},
    {"id": 10, "text": "Apakah selama 30 hari terakhir ini Anda menangis lebih sering?", "cat": "Depresi / Emosional"},
    {"id": 11, "text": "Apakah selama 30 hari terakhir ini Anda merasa sulit untuk menikmati kegiatan sehari-hari?", "cat": "Anhedonia / Depresi"},
    {"id": 12, "text": "Apakah selama 30 hari terakhir ini Anda sulit untuk mengambil keputusan?", "cat": "Kognitif"},
    {"id": 13, "text": "Apakah selama 30 hari terakhir ini pekerjaan sehari-hari Anda terganggu?", "cat": "Fungsional"},
    {"id": 14, "text": "Apakah selama 30 hari terakhir ini Anda tidak mampu melakukan hal-hal yang bermanfaat dalam hidup Anda?", "cat": "Fungsional / Depresi"},
    {"id": 15, "text": "Apakah selama 30 hari terakhir ini Anda kehilangan minat pada berbagai hal?", "cat": "Anhedonia / Depresi"},
    {"id": 16, "text": "Apakah selama 30 hari terakhir ini Anda merasa tidak berharga?", "cat": "Harga Diri / Depresi"},
    {"id": 17, "text": "Apakah selama 30 hari terakhir ini Anda mempunyai pikiran untuk mengakhiri hidup?", "cat": "Kritis / Suisidal"},
    {"id": 18, "text": "Apakah selama 30 hari terakhir ini Anda merasa lelah sepanjang waktu?", "cat": "Energi / Somatik"},
    {"id": 19, "text": "Apakah selama 30 hari terakhir ini Anda mengalami rasa tidak enak di perut?", "cat": "Fisik / Somatik"},
    {"id": 20, "text": "Apakah selama 30 hari terakhir ini Anda mudah lelah?", "cat": "Energi / Somatik"}
]

# ==========================================
# GENERATOR DATASET BAWAAN (214 RESPONDEN)
# ==========================================
@st.cache_data
def generate_default_data():
    np.random.seed(42)
    fakultas_list = ["Fakultas Ilmu Komputer", "Fakultas Ekonomi & Bisnis", "Fakultas Teknik", "Fakultas Kedokteran", "Fakultas Ilmu Sosial & Politik"]
    records = []
    
    for i in range(1, 160):
        skor = np.random.randint(0, 6)
        jawaban = [0] * 20
        idx_ya = np.random.choice(20, skor, replace=False)
        for idx in idx_ya:
            jawaban[idx] = 1
        records.append([f"Responden Mahasiswa #{i}", fakultas_list[i % len(fakultas_list)]] + jawaban + [skor, 0])
        
    for i in range(160, 215):
        skor = np.random.randint(6, 16)
        jawaban = [0] * 20
        idx_ya = np.random.choice(20, skor, replace=False)
        for idx in idx_ya:
            jawaban[idx] = 1
        records.append([f"Responden Mahasiswa #{i}", fakultas_list[i % len(fakultas_list)]] + jawaban + [skor, 1])
        
    kolom = ["Nama Responden / ID", "Fakultas / Prodi"] + [f"Q{i}" for i in range(1, 21)] + ["Score", "Kategori"]
    return pd.DataFrame(records, columns=kolom)

# ==========================================
# JALUR PIPELINE TRAINING MODEL AI
# ==========================================
def train_naive_bayes(dataframe):
    fitur = [f"Q{i}" for i in range(1, 21)]
    X = dataframe[fitur]
    y = dataframe["Kategori"]
    
    if dataframe["Kategori"].value_counts().min() < 2:
        return None, 0, 0, 0, 0
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = BernoulliNB(alpha=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "acc": accuracy_score(y_test, y_pred) * 100,
        "prec": precision_score(y_test, y_pred, zero_division=0) * 100,
        "rec": recall_score(y_test, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_test, y_pred, zero_division=0) * 100
    }
    return model, metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"]

# ==========================================
# LAYOUT HEADER UTAMA
# ==========================================
st.markdown("<h2 style='margin-bottom:0px;'>🧠 SRQ-20 Screening System</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:gray; font-size:1.1rem;'>Self-Reporting Questionnaire & Naive Bayes AI Analytics</p>", unsafe_allow_html=True)
st.divider()

tab_screening, tab_analytics, tab_about = st.tabs([
    "📋 Kuesioner Interaktif", 
    "📊 Dashboard AI & Dataset", 
    "ℹ️ Panduan & Edukasi"
])

# ==========================================
# TAB 1: KUESIONER INTERAKTIF
# ==========================================
with tab_screening:
    st.subheader("Skrining Kesehatan Mental Mandiri")
    st.write("Silakan isi formulir identitas opsional di bawah ini dan jawab 20 pertanyaan kuesioner sesuai dengan kondisi Anda dalam **30 hari terakhir**.")
    
    c_bio1, c_bio2, c_bio3 = st.columns(3)
    with c_bio1: nama_input = st.text_input("Nama Lengkap / Inisial", placeholder="Contoh: Mahasiswa A")
    with c_bio2: instansi_input = st.text_input("Fakultas / Instansi", placeholder="Contoh: Fakultas Ilmu Komputer")
    with c_bio3: usia_input = st.text_input("Semester / Usia", placeholder="Contoh: Semester 4 / 20 Tahun")
    
    st.markdown("---")
    
    with st.form(key="form_kuesioner_srq"):
        jawaban_user = []
        col_kiri, col_kanan = st.columns(2)
        
        for idx, q in enumerate(srq_questions):
            posisi_kolom = col_kiri if idx < 10 else col_kanan
            with posisi_kolom:
                pilihan = st.radio(
                    f"**#{q['id']}** {q['text']}",
                    options=["Tidak", "Ya"],
                    key=f"pertanyaan_{q['id']}",
                    horizontal=True
                )
                jawaban_user.append(1 if pilihan == "Ya" else 0)
                
        submit_form = st.form_submit_button("✨ Lihat Hasil Evaluasi SRQ-20")
        
    if submit_form:
        total_skor = sum(jawaban_user)
        terindikasi = total_skor >= 6
        
        st.markdown("---")
        st.subheader("🏆 Laporan Hasil Skrining Kesehatan Mental")
        st.write(f"**Skor Anda: {total_skor} dari 20**")
        
        if terindikasi:
            st.markdown('<div class="status-badge badge-stress">⚠️ TERINDIKASI DISTRES PSIKOLOGIS (KATEGORI 1)</div>', unsafe_allow_html=True)
            st.error("Melampaui batas ambang/Cut-off score WHO ≥ 6. Skor ini menunjukkan Anda sedang mengalami indikasi ketegangan emosional atau stres psikologis.")
            st.markdown("""
            **💡 Rekomendasi Tindakan Lanjutan:**
            * Batasi beban tugas kognitif berlebih dan atur waktu istirahat.
            * Berceritalah dengan teman dekat atau layanan konselor sebaya kampus.
            * **Sangat Disarankan:** Konsultasi langsung dengan konselor BK atau Psikolog Profesional.
            """)
        else:
            st.markdown('<div class="status-badge badge-normal">✅ TIDAK TERINDIKASI / KONDISI STABIL (KATEGORI 0)</div>', unsafe_allow_html=True)
            st.success("Di bawah batas ambang/Cut-off score WHO < 6. Kondisi psikologis dan emosional Anda saat ini tergolong stabil.")
            
        st.subheader("📝 Rincian Gejala Yang Dilaporkan")
        if total_skor > 0:
            for idx, val in enumerate(jawaban_user):
                if val == 1:
                    st.markdown(f"* **Q{idx+1}:** {srq_questions[idx]['text']}")
        else:
            st.info("Luar biasa! Tidak ada keluhan gejala yang dilaporkan.")

# ==========================================
# TAB 2: DASHBOARD AI & DATASET
# ==========================================
with tab_analytics:
    st.subheader("📈 Analisis Dataset & Model Klasifikasi")
    
    uploaded_file = st.file_uploader("📁 Upload Dataset Custom Anda (.xlsx atau .csv)", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_data = pd.read_csv(uploaded_file)
            else:
                df_data = pd.read_excel(uploaded_file)
                
            if 'Q1' not in df_data.columns and len(df_data.columns) >= 25:
                rename_dict = {df_data.columns[i+5]: f'Q{i+1}' for i in range(20)}
                df_data.rename(columns=rename_dict, inplace=True)
            elif 'Q1' not in df_data.columns and len(df_data.columns) >= 20:
                rename_dict = {df_data.columns[i]: f'Q{i+1}' for i in range(20)}
                df_data.rename(columns=rename_dict, inplace=True)
                
            kolom_fitur = [f"Q{i}" for i in range(1, 21)]
            for col in kolom_fitur:
                if col in df_data.columns:
                    df_data[col] = df_data[col].replace({'Ya': 1, 'Tidak': 0, '1': 1, '0': 0}).astype(int)
            
            if "Score" not in df_data.columns:
                df_data["Score"] = df_data[kolom_fitur].sum(axis=1)
            if "Kategori" not in df_data.columns:
                df_data["Kategori"] = np.where(df_data["Score"] >= 6, 1, 0)
                
            st.success(f"Dataset kustom '{uploaded_file.name}' berhasil dimuat!")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}. Menggunakan data simulasi bawaan.")
            df_data = generate_default_data()
    else:
        df_data = generate_default_data()
        
    total_responden = len(df_data)
    total_normal = len(df_data[df_data["Kategori"] == 0])
    total_stres = len(df_data[df_data["Kategori"] == 1])
    
    model_nb, acc, prec, rec, f1 = train_naive_bayes(df_data)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Responden", f"{total_responden} Orang")
    m2.metric("🟢 Kategori Stabil (< 6)", f"{total_normal}")
    m3.metric("🔴 Terindikasi Stres (≥ 6)", f"{total_stres}")
    m4.metric("🎯 Akurasi BernoulliNB", f"{round(acc, 1)}%")
    
    st.markdown("---")
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.write("#### 📊 Grafik Distribusi Kategori SRQ-20")
        fig_pie = px.pie(
            names=["Kategori Stabil (0)", "Terindikasi Stres (1)"],
            values=[total_normal, total_stres],
            color_discrete_sequence=["#10B981", "#F43F5E"],
            hole=0.4
        )
        st.plotly_chart(fig_pie)
        
    with g_col2:
        st.write("#### 🎯 Metrik Evaluasi Performa Model AI (%)")
        fig_bar = px.bar(
            x=["Akurasi", "Precision", "Recall", "F1-Score"],
            y=[acc, prec, rec, f1],
            text=[f"{round(acc,1)}%", f"{round(prec,1)}%", f"{round(rec,1)}%", f"{round(f1,1)}%"],
            color=["Akurasi", "Precision", "Recall", "F1-Score"],
            color_discrete_sequence=["#6366F1", "#8B5CF6", "#3B82F6", "#EC4899"]
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, yaxis=dict(range=[0, 115], title="Nilai Persen (%)"), xaxis=dict(title=""))
        st.plotly_chart(fig_bar)
        
    st.markdown("---")
    
    st.write("#### 📑 Pratinjau Tabel Data Responden")
    opsi_filter = st.selectbox("Saring Tampilan Berdasarkan Kategori:", ["Semua Data", "🔴 Hanya Terindikasi Stres (≥ 6)", "🟢 Hanya Kategori Stabil (< 6)"])
    
    if opsi_filter == "🔴 Hanya Terindikasi Stres (≥ 6)":
        df_tampil = df_data[df_data["Kategori"] == 1]
    elif opsi_filter == "🟢 Hanya Kategori Stabil (< 6)":
        df_tampil = df_data[df_data["Kategori"] == 0]
    else:
        df_tampil = df_data
        
    st.dataframe(df_tampil, hide_index=True)

# ==========================================
# TAB 3: PANDUAN & EDUKASI
# ==========================================
with tab_about:
    st.subheader("Mengenal Instrumen WHO Self-Reporting Questionnaire (SRQ-20)")
    st.write("""
    **SRQ-20** adalah instrumen resmi yang dikembangkan oleh organisasi kesehatan dunia (**WHO**) untuk mendeteksi dini masalah gangguan neurotik/psikologis minor secara massal, khususnya di negara berkembang. 
    """)
    
    st.markdown("---")
    
    c_box1, c_box2 = st.columns(2)
    with c_box1:
        st.info("**Skor Total 0 s.d 5 : Kategori Stabil (0)**\nMenandakan kondisi mental responden cenderung normal dan sehat. Indikasi stres harian dinilai masih wajar.")
    with c_box2:
        st.error("**Skor Total ≥ 6 : Terindikasi Distres (1)**\nMenandakan potensi indikasi masalah emosional ringan hingga berat. Responden disarankan mencari validasi profesional.")
