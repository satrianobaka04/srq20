import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="SRQ-20 Mental Health Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS dasar untuk menyamai tema gelap
# st.markdown("""
#     <style>
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#     }
#     .stTabs [data-baseweb="tab"] {
#         background-color: transparent;
#         border-radius: 8px;
#         padding: 10px 20px;
#         border: 1px solid rgba(255, 255, 255, 0.1);
#     }
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #6366F1, #8B5CF6);
#         color: white;
#         border: none;
#     }
#     .status-badge {
#         padding: 10px 20px;
#         border-radius: 10px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     .badge-normal {
#         background-color: rgba(16, 185, 129, 0.2);
#         color: #34D399;
#         border: 1px solid #10B981;
#     }
#     .badge-stress {
#         background-color: rgba(244, 63, 94, 0.2);
#         color: #FB7185;
#         border: 1px solid #F43F5E;
#     }
#     </style>
# """, unsafe_allow_html=True)

# ==========================================
# DATA PERTANYAAN SRQ-20
# ==========================================
srq_questions = [
    "Apakah selama 30 hari terakhir ini Anda sering menderita sakit kepala?",
    "Apakah selama 30 hari terakhir ini Anda tidak nafsu makan?",
    "Apakah selama 30 hari terakhir ini Anda sulit tidur?",
    "Apakah selama 30 hari terakhir ini Anda mudah takut?",
    "Apakah selama 30 hari terakhir Anda merasa tegang, cemas, atau kuatir?",
    "Apakah selama 30 hari terakhir ini tangan Anda gemetar?",
    "Apakah selama 30 hari terakhir ini pencernaan Anda terganggu / buruk?",
    "Apakah selama 30 hari terakhir ini Anda sulit berpikir jernih?",
    "Apakah selama 30 hari terakhir ini Anda merasa tidak bahagia?",
    "Apakah selama 30 hari terakhir ini Anda menangis lebih sering?",
    "Apakah selama 30 hari terakhir ini Anda merasa sulit untuk menikmati kegiatan sehari-hari?",
    "Apakah selama 30 hari terakhir ini Anda sulit untuk mengambil keputusan?",
    "Apakah selama 30 hari terakhir ini pekerjaan sehari-hari Anda terganggu?",
    "Apakah selama 30 hari terakhir ini Anda tidak mampu melakukan hal-hal yang bermanfaat dalam hidup Anda?",
    "Apakah selama 30 hari terakhir ini Anda kehilangan minat pada berbagai hal?",
    "Apakah selama 30 hari terakhir ini Anda merasa tidak berharga?",
    "Apakah selama 30 hari terakhir ini Anda mempunyai pikiran untuk mengakhiri hidup?",
    "Apakah selama 30 hari terakhir ini Anda merasa lelah sepanjang waktu?",
    "Apakah selama 30 hari terakhir ini Anda mengalami rasa tidak enak di perut?",
    "Apakah selama 30 hari terakhir ini Anda mudah lelah?"
]

# ==========================================
# FUNGSI GENERATE DEFAULT DATASET (Menyamai JS)
# ==========================================
@st.cache_data
def generate_default_dataset():
    fakultas = ["Ilmu Komputer", "Ekonomi & Bisnis", "Teknik", "Kedokteran", "Ilmu Sosial & Politik"]
    records = []
    
    # 159 Normal (Skor < 6)
    for i in range(1, 160):
        skor = np.random.randint(0, 6)
        answers = [0] * 20
        indices = np.random.choice(20, skor, replace=False)
        for idx in indices:
            answers[idx] = 1
            
        record = [f"Responden #{i}", fakultas[i % len(fakultas)]] + answers + [skor, 0]
        records.append(record)
        
    # 55 Terindikasi (Skor >= 6)
    for i in range(160, 215):
        skor = np.random.randint(6, 16)
        answers = [0] * 20
        indices = np.random.choice(20, skor, replace=False)
        for idx in indices:
            answers[idx] = 1
            
        record = [f"Responden #{i}", fakultas[i % len(fakultas)]] + answers + [skor, 1]
        records.append(record)
        
    kolom = ['Nama', 'Fakultas'] + [f'Q{i}' for i in range(1, 21)] + ['Score', 'Kategori']
    return pd.DataFrame(records, columns=kolom)

# ==========================================
# HEADER
# ==========================================
col1, col2 = st.columns([1, 11])
with col1:
    st.markdown("<h1 style='text-align: center;'>🧠</h1>", unsafe_allow_html=True)
with col2:
    st.title("SRQ-20 Screening System")
    st.markdown("Self-Reporting Questionnaire & Naive Bayes AI Analytics")

st.divider()

# ==========================================
# TABS NAVIGASI
# ==========================================
tab_screening, tab_analytics, tab_about = st.tabs([
    "📋 Kuesioner Interaktif", 
    "📊 Dashboard AI & Dataset", 
    "ℹ️ Panduan & Edukasi"
])

# ==========================================
# TAB 1: KUESIONER INTERAKTIF
# ==========================================
with tab_screening:
    st.header("Skrining Kesehatan Mental SRQ-20")
    st.info("Jawablah 20 pertanyaan di bawah ini sesuai dengan kondisi yang Anda rasakan atau alami dalam **30 hari terakhir**.")
    
    # Identitas
    st.subheader("👤 Identitas Responden (Opsional)")
    col_bio1, col_bio2, col_bio3 = st.columns(3)
    with col_bio1: nama = st.text_input("Nama Lengkap / Inisial")
    with col_bio2: instansi = st.text_input("Fakultas / Instansi")
    with col_bio3: usia = st.text_input("Semester / Usia")
    
    st.markdown("---")
    
    # Kuesioner Form
    with st.form("srq_form"):
        jawaban_user = []
        col_q1, col_q2 = st.columns(2)
        
        for i, pertanyaan in enumerate(srq_questions):
            target_col = col_q1 if i < 10 else col_q2
            with target_col:
                jawaban = st.radio(
                    f"**Q{i+1}:** {pertanyaan}", 
                    options=["Tidak", "Ya"], 
                    key=f"q_{i}",
                    horizontal=True
                )
                jawaban_user.append(1 if jawaban == "Ya" else 0)
                
        submit_btn = st.form_submit_button("✨ Lihat Hasil Evaluasi SRQ-20", use_container_width=True)
        
    # Logika Penilaian
    if submit_btn:
        total_skor = sum(jawaban_user)
        is_indicated = total_skor >= 6
        
        st.markdown("---")
        st.subheader("🏆 Laporan Hasil Skrining Kesehatan Mental")
        
        if is_indicated:
            st.markdown(f'<div class="status-badge badge-stress">⚠️ TERINDIKASI DISTRES PSIKOLOGIS (KATEGORI 1)</div>', unsafe_allow_html=True)
            st.error(f"**Skor Anda: {total_skor} / 20** (Melampaui batas ambang/Cut-off score WHO ≥ 6). Hal ini menunjukkan adanya beban emosional atau stres yang cukup tinggi.")
            st.markdown("""
            **💡 Rekomendasi Tindakan Lanjutan:**
            * Disarankan untuk mengatur jadwal istirahat dan mengurangi beban kognitif berlebih.
            * Luangkan waktu untuk berbagi cerita dengan kerabat terpercaya atau konselor sebaya.
            * **Sangat dianjurkan:** Mengunjungi layanan Bimbingan Konseling Kampus atau Psikolog Profesional untuk pendampingan lebih lanjut.
            """)
        else:
            st.markdown(f'<div class="status-badge badge-normal">✅ TIDAK TERINDIKASI / KONDISI STABIL (KATEGORI 0)</div>', unsafe_allow_html=True)
            st.success(f"**Skor Anda: {total_skor} / 20** (Di bawah batas ambang/Cut-off score WHO < 6). Kondisi kesehatan mental dan emosional Anda tergolong stabil dan positif.")
            st.markdown("""
            **🌱 Tips Mempertahankan Kesehatan Mental:**
            * Pertahankan pola tidur yang teratur dan asupan nutrisi seimbang.
            * Lanjutkan aktivitas fisik/olahraga ringan dan hobi yang menyenangkan.
            * Tetap peka terhadap kondisi emosional diri sendiri dan rekan di sekitar Anda.
            """)
            
        # Tampilkan Gejala
        if total_skor > 0:
            st.write("📝 **Gejala yang Dilaporkan:**")
            for i, val in enumerate(jawaban_user):
                if val == 1:
                    st.markdown(f"- Q{i+1}: {srq_questions[i]}")
        else:
            st.write("🌟 Luar biasa! Tidak ada keluhan gejala yang dilaporkan.")

# ==========================================
# TAB 2: DASHBOARD AI & DATASET
# ==========================================
# ==========================================
# TAB 2: DASHBOARD AI & DATASET
# ==========================================
with tab_analytics:
    st.subheader("📈 Analisis Dataset & Model Klasifikasi")
    
    # Fitur unggah berkas kustom
    uploaded_file = st.file_uploader("📁 Upload Dataset Custom Anda (.xlsx atau .csv)", type=["xlsx", "csv"])
    
    # Penentuan sumber data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_data = pd.read_csv(uploaded_file)
            else:
                df_data = pd.read_excel(uploaded_file)
                
            # --- FIX: GANTI NAMA KOLOM MENJADI Q1-Q20 ---
            # Jika dataset asli diupload, kita ubah kolom ke-6 hingga ke-25 menjadi Q1-Q20
            # Sesuai dengan "TAHAP 3 : SELECTION DATA" di script asli Anda
            if 'Q1' not in df_data.columns and len(df_data.columns) >= 25:
                rename_dict = {df_data.columns[i+5]: f'Q{i+1}' for i in range(20)}
                df_data.rename(columns=rename_dict, inplace=True)
            elif 'Q1' not in df_data.columns and len(df_data.columns) >= 20:
                # Fallback jika kolom kurang dari 25 tapi minimal ada 20
                rename_dict = {df_data.columns[i]: f'Q{i+1}' for i in range(20)}
                df_data.rename(columns=rename_dict, inplace=True)
                
            # Pembersihan kolom otomatis (mencari kolom Q1-Q20)
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
        
    # Kalkulasi Variabel & Metrik Klasifikasi
    total_responden = len(df_data)
    total_normal = len(df_data[df_data["Kategori"] == 0])
    total_stres = len(df_data[df_data["Kategori"] == 1])
    
    model_nb, acc, prec, rec, f1 = train_naive_bayes(df_data)
    
    # Tampilan KPI Stat Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Responden", f"{total_responden} Orang")
    m2.metric("🟢 Kategori Stabil (< 6)", f"{total_normal} ({round(total_normal/total_responden*100, 1)}%)")
    m3.metric("🔴 Terindikasi Stres (≥ 6)", f"{total_stres} ({round(total_stres/total_responden*100, 1)}%)")
    m4.metric("🎯 Akurasi BernoulliNB", f"{round(acc, 1)}%")
    
    st.markdown("---")
    
    # Sesi Grafik Visualisasi dengan Plotly
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.write("#### 📊 Grafik Distribusi Kategori SRQ-20")
        fig_pie = px.pie(
            names=["Kategori Stabil (0)", "Terindikasi Stres (1)"],
            values=[total_normal, total_stres],
            color_discrete_sequence=["#10B981", "#F43F5E"],
            hole=0.4
        )
        # Mengganti use_container_width menjadi width="stretch" (Fix warning)
        st.plotly_chart(fig_pie, width="stretch")
        
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
        # Mengganti use_container_width menjadi width="stretch" (Fix warning)
        st.plotly_chart(fig_bar, width="stretch")
        
    st.markdown("---")
    
    # Tabel Pratinjau Dataset Berdasarkan Filter Kategori
    st.write("#### 📑 Pratinjau Tabel Data Responden")
    opsi_filter = st.selectbox("Saring Tampilan Berdasarkan Kategori:", ["Semua Data", "🔴 Hanya Terindikasi Stres (≥ 6)", "🟢 Hanya Kategori Stabil (< 6)"])
    
    if opsi_filter == "🔴 Hanya Terindikasi Stres (≥ 6)":
        df_tampil = df_data[df_data["Kategori"] == 1]
    elif opsi_filter == "🟢 Hanya Kategori Stabil (< 6)":
        df_tampil = df_data[df_data["Kategori"] == 0]
    else:
        df_tampil = df_data
        
    # Mengganti use_container_width menjadi width="stretch" (Fix warning)
    st.dataframe(df_tampil, width="stretch", hide_index=True)

# ==========================================
# TAB 3: PANDUAN & EDUKASI
# ==========================================
with tab_about:
    st.header("🌍 Tentang Self-Reporting Questionnaire (SRQ-20)")
    st.write("""
    **SRQ-20** adalah instrumen skrining gangguan kesehatan mental yang dikembangkan oleh **World Health Organization (WHO)** khususnya untuk negara berkembang. Instrumen ini terdiri dari 20 pertanyaan sederhana dengan jawaban "Ya" atau "Tidak" mengenai gejala non-psikotik yang dialami dalam 30 hari terakhir.
    """)
    
    st.markdown("---")
    
    st.subheader("💡 Aturan Penilaian & Cut-Off Score")
    col_rule1, col_rule2 = st.columns(2)
    with col_rule1:
        st.info("""
        **Skor 0 - 5 : Tidak Terindikasi**\n
        Kondisi mental dan emosional secara umum stabil. Gejala stres harian berada pada batas wajar yang dapat diatasi dengan mekanisme koping standar.
        """)
    with col_rule2:
        st.error("""
        **Skor ≥ 6 : Terindikasi Distres Psikologis**\n
        Menunjukkan adanya potensi masalah emosional seperti kecemasan, stres berat, atau gejala depresi ringan. Disarankan untuk berkonsultasi dengan profesional (psikolog/konselor).
        """)

    st.markdown("---")
    
    st.subheader("🛡️ Catatan Penting & Etika Skrining")
    st.markdown("""
    * **Bukan Diagnosis Mutlak:** Hasil kuesioner ini merupakan skrining awal (deteksi dini), bukan diagnosis klinis resmi.
    * **Privasi Terjamin:** Seluruh data kuesioner yang Anda isi pada aplikasi ini tidak disimpan ke dalam database. Data langsung hilang saat halaman di-*refresh*.
    * **Bantuan Darurat:** Apabila Anda merasa dalam krisis atau memiliki pikiran untuk membahayakan diri sendiri, segera hubungi fasilitas kesehatan terdekat atau layanan konseling profesional.
    """)
