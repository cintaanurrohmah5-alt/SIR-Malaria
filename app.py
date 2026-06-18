import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# ==========================================
# 1. KONFIGURASI HALAMAN & DATABASE WILAYAH
# ==========================================
st.set_page_config(page_title="Dashboard Simulasi Malaria", layout="wide")

# DATABASE PARAMETER UNTUK 8 KABUPATEN PAPUA TENGAH
DATABASE_WILAYAH = {
    "Kabupaten Mimika": {
        "Nh": 6361, "k_base": 7000, "alpha": 350.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Mimika.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Mimika_BPJS.csv")
    },
    "Kabupaten Nabire": {
        "Nh": 14948, "k_base": 15000, "alpha": 750.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Nabire.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Nabire_BPJS.csv")
    },
    "Kabupaten Dogiyai": {
        "Nh": 2190, "k_base": 4000, "alpha": 200.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Dogiyai.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Dogiyai_BPJS.csv")
    },
    "Kabupaten Deiyai": {
        "Nh": 5072, "k_base": 6000, "alpha": 300.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Deiyai.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Deiyai_BPJS.csv")
    },
    "Kabupaten Paniai": {
        "Nh": 12591, "k_base": 13000, "alpha": 650.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Paniai.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Paniai_BPJS.csv")
    },
    "Kabupaten Intan Jaya": {
        "Nh": 1037, "k_base": 1500, "alpha": 75.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Intan Jaya.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Intan Jaya_BPJS.csv")
    },
    "Kabupaten Puncak Jaya": {
        "Nh": 11537, "k_base": 12000, "alpha": 600.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Puncak Jaya.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Puncak Jaya_BPJS.csv")
    },
    "Kabupaten Puncak": {
        "Nh": 1075, "k_base": 1500, "alpha": 75.0, "gap_hari": 30,
        "file_iklim": os.path.join("data", "Kabupaten Puncak.csv"), 
        "file_bpjs": os.path.join("data", "Malaria_Kabupaten Puncak_BPJS.csv")
    },
    "Unggah Data Mandiri (Wilayah Lain)": {
        "Nh": 1000000, "k_base": 15000, "alpha": 350.0, "gap_hari": 30,
        "file_iklim": None, "file_bpjs": None
    }
}

st.title("📊 Sistem Simulasi Malaria (Model SIR)")
st.markdown("Aplikasi interaktif pemodelan dinamika malaria berbasis iklim untuk Provinsi Papua Tengah & Wilayah Universal.")

with st.expander("Pelajari Model Matematika SIR yang Digunakan"):
    st.markdown("Sistem ini menggunakan gabungan model kompartemen **SIR** (untuk dinamika populasi manusia) dan **LM** (untuk siklus hidup akuatik vektor nyamuk Anopheles). Sistem ini bersifat *non-autonomous* karena nilai parameternya terus berubah mengikuti fluktuasi iklim harian.")
    
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("**1. Sub-Model Nyamuk (Vektor)**")
        st.latex(r''' \frac{dA}{dt} = \phi (S_v + I_v)\left(1 - \frac{A}{K_R(t)}\right) - (\sigma + \mu_A)A ''')
        st.latex(r''' \frac{dS_v}{dt} = \sigma A - \beta_{hv}(t) S_v \frac{I_h}{N_h} - \mu_v(t) S_v ''')
        st.latex(r''' \frac{dI_v}{dt} = \beta_{hv}(t) S_v \frac{I_h}{N_h} - \mu_v(t) I_v ''')
        
    with col_eq2:
        st.markdown("**2. Sub-Model Manusia (Inang)**")
        st.latex(r''' \frac{dS_h}{dt} = \Lambda_h - \beta_{vh}(t) S_h \frac{I_v}{N_h} - \mu_h S_h + \omega R_h ''')
        st.latex(r''' \frac{dI_h}{dt} = \beta_{vh}(t) S_h \frac{I_v}{N_h} - (r + \mu_h + \delta) I_h ''')
        st.latex(r''' \frac{dR_h}{dt} = r I_h - \mu_h R_h - \omega R_h ''')

    st.markdown("**Keterangan Variabel & Parameter Dinamis:**")
    st.markdown("""
    * **$A, S_v, I_v$**: Berturut-turut adalah fase akuatik (jentik), nyamuk dewasa rentan, dan nyamuk terinfeksi.
    * **$S_h, I_h, R_h$**: Berturut-turut adalah manusia rentan, terinfeksi, dan sembuh.
    * **$K_R(t)$**: Kapasitas daya tampung lingkungan untuk jentik, dievaluasi berdasarkan akumulasi curah hujan.
    * **$\\beta_{vh}(t)$ & $\\beta_{hv}(t)$**: Laju penularan parasit dari nyamuk ke manusia dan manusia ke nyamuk. Dimodelkan menggunakan *Fungsi Brière* untuk menyesuaikan sensitivitas aktivitas vektor terhadap suhu harian.
    * **$\\mu_v(t)$**: Laju kematian nyamuk dewasa yang bervariasi bergantung pada suhu ekstrem lingkungan.
    * **$\\phi, \\sigma, \\mu_A$**: Tingkat bertelur nyamuk, laju transisi dari fase akuatik menjadi dewasa, dan laju kematian alami fase akuatik.
    * **$\\Lambda_h, \\mu_h, \\delta, r, \\omega$**: Laju kelahiran manusia, kematian alami manusia, kematian spesifik akibat malaria, laju kesembuhan, dan tingkat hilangnya kekebalan.
    """)

# ==========================================
# 2. SIDEBAR: DROPDOWN & PENGATURAN
# ==========================================
st.sidebar.header("Konfigurasi Wilayah")

pilihan_wilayah = st.sidebar.selectbox(
    "Pilih Wilayah Analisis:", 
    options=list(DATABASE_WILAYAH.keys())
)

data_default = DATABASE_WILAYAH[pilihan_wilayah]
is_mandiri = (pilihan_wilayah == "Unggah Data Mandiri (Wilayah Lain)")

with st.sidebar.expander("Parameter Demografi & Vektor", expanded=True):
    Nh_input = st.number_input("Total Populasi Penduduk:", value=data_default["Nh"], step=100)
    k_base = st.number_input("K_min (Daya Tampung Dasar)", min_value=100, value=int(data_default["k_base"]), step=500)
    alpha = st.number_input("Theta (Faktor Konversi Hujan)", min_value=10.0, value=data_default["alpha"], step=10.0)
    if not is_mandiri:
        gap_hari = st.number_input("Deduplikasi Episode BPJS (Hari)", min_value=7, value=data_default["gap_hari"], step=1)

# ==========================================
# 3. KENDALI INPUT DATA (OTOMATIS VS MANUAL)
# ==========================================
df_bpjs = None
df_iklim = None

if is_mandiri:
    st.markdown("### 📥 Unggah Dataset Wilayah Mandiri")
    st.info("Mode Mandiri: Anda cukup mengunggah Data Iklim NASA (Suhu & Curah Hujan). Data BPJS tidak diperlukan.")
    file_iklim_upload = st.file_uploader("Unggah Data Iklim NASA (Format CSV)", type=["csv"])
    
    if file_iklim_upload is None:
        st.stop()
    else:
        file_iklim_source = file_iklim_upload
else:
    st.markdown(f"### Mode Presentasi: **{pilihan_wilayah}**")
    st.success(f"Menggunakan parameter spesifik dan dataset internal terenkripsi untuk {pilihan_wilayah}.")
    
    if not os.path.exists(data_default["file_iklim"]) or not os.path.exists(data_default["file_bpjs"]):
        st.error(f"File data untuk **{pilihan_wilayah}** tidak ditemukan di dalam folder data.")
        st.warning(f"Pastikan folder bernama `data` berisi file dengan nama berikut:\n- Iklim: `{os.path.basename(data_default['file_iklim'])}`\n- BPJS: `{os.path.basename(data_default['file_bpjs'])}`")
        st.stop()
        
    file_iklim_source = data_default["file_iklim"]
    file_bpjs_source = data_default["file_bpjs"]

# ==========================================
# 4. PROSES DATA & PEMODELAN MATEMATIKA
# ==========================================
try:
    with st.spinner("Memproses data dan mengkalkulasi model SIR-LM..."):
        
        # --- A. PROSES DATA BPJS ---
        if not is_mandiri:
            df_bpjs = pd.read_csv(file_bpjs_source, sep=None, engine='python')
            df_bpjs.columns = df_bpjs.columns.str.strip()
            df_bpjs['Tgl_Start'] = pd.to_datetime(df_bpjs['Tgl_Start'], dayfirst=True, errors='coerce')
            df_bpjs = df_bpjs.dropna(subset=['Tgl_Start'])
            df_bpjs.sort_values(by=['ID_Member', 'Tgl_Start'], inplace=True)
            
            def filter_episode(group, gap):
                keep = []
                last_episode_start = None
                for date in group['Tgl_Start']:
                    if last_episode_start is None or (date - last_episode_start).days > gap:
                        keep.append(True)
                        last_episode_start = date
                    else:
                        keep.append(False)
                return keep

            mask = df_bpjs.groupby('ID_Member').apply(lambda x: filter_episode(x, gap_hari)).explode().values
            df_bpjs_unik = df_bpjs[mask.astype(bool)].copy()
            df_bpjs_mingguan = df_bpjs_unik.resample('W-MON', on='Tgl_Start')['Bobot'].sum().reset_index()
            df_bpjs_mingguan.rename(columns={'Tgl_Start': 'Tanggal_Minggu', 'Bobot': 'Insiden_Kasus'}, inplace=True)
            df_bpjs_mingguan.set_index('Tanggal_Minggu', inplace=True)

        # --- B. PROSES DATA IKLIM NASA ---
        if is_mandiri:
            lines = [line.decode('utf-8') for line in file_iklim_upload.readlines()]
            file_iklim_upload.seek(0)
        else:
            with open(file_iklim_source, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        
        skip_rows = 0
        for i, line in enumerate(lines):
            if 'YEAR' in line and 'DOY' in line:
                skip_rows = i
                break
                
        if is_mandiri:
            df_iklim = pd.read_csv(file_iklim_source, skiprows=skip_rows, sep=None, engine='python')
        else:
            df_iklim = pd.read_csv(file_iklim_source, skiprows=skip_rows, sep=None, engine='python')
                
        df_iklim.columns = df_iklim.columns.str.strip()
        df_iklim.replace(-999.0, np.nan, inplace=True)
        df_iklim['Date'] = pd.to_datetime(df_iklim['YEAR'] * 1000 + df_iklim['DOY'], format='%Y%j')
        df_iklim.set_index('Date', inplace=True)
        df_iklim['T2M'] = df_iklim['T2M'].interpolate(method='linear').fillna(24.0)
        df_iklim['PRECTOTCORR'] = df_iklim['PRECTOTCORR'].fillna(0)

        threshold_hujan = 150.0
        df_iklim['PRECTOTCORR_ROLL14'] = df_iklim['PRECTOTCORR'].clip(upper=threshold_hujan).rolling(window=14).sum().shift(1)
        df_iklim.dropna(subset=['PRECTOTCORR_ROLL14'], inplace=True)
        indeks_waktu = np.arange(len(df_iklim))
        tanggal_iklim = df_iklim.index

        # --- C. SISTEM ODE (SIR-LM MODEL) ---
        c_briere = 0.000116; T_min_b = 15.64; T_max_b = 34.92; p_hv = 0.75; p_vh = 0.5
        
        def briere_function(T):
            return np.where((T > T_min_b) & (T < T_max_b), c_briere * T * (T - T_min_b) * np.sqrt(T_max_b - T), 0)

        df_iklim['BETA_hv'] = briere_function(df_iklim['T2M']) * p_hv * 3.5
        df_iklim['BETA_vh'] = briere_function(df_iklim['T2M']) * p_vh * 3.5
        df_iklim['MU_v'] = np.minimum(0.002 * (df_iklim['T2M'] - 35.0)**2 + 0.05, 1.0)
        df_iklim['K_R'] = k_base + alpha * df_iklim['PRECTOTCORR_ROLL14']

        K_func = interp1d(indeks_waktu, df_iklim['K_R'].values, kind='linear', fill_value="extrapolate")
        Beta_hv_func = interp1d(indeks_waktu, df_iklim['BETA_hv'].values, kind='linear', fill_value="extrapolate")
        Beta_vh_func = interp1d(indeks_waktu, df_iklim['BETA_vh'].values, kind='linear', fill_value="extrapolate")
        MU_v_func = interp1d(indeks_waktu, df_iklim['MU_v'].values, kind='linear', fill_value="extrapolate")

        Nh = Nh_input
        mu_h = 1 / (66.44 * 365); Lambda_h = (0.0223 * Nh) / 365
        omega = 1 / 120; r = 1 / 14; delta = 0.01; phi = 3.8; sigma = 0.07; mu_A = 0.32 

        def sirlm_model(y, t):
            A, Sv, Iv, Sh, Ih, Rh = y
            K_R_t = max(K_func(t), 1.0)
            beta_hv_t = max(Beta_hv_func(t), 0.0); beta_vh_t = max(Beta_vh_func(t), 0.0); mu_v_t = max(MU_v_func(t), 0.0)
            
            dA_dt = phi * (Sv + Iv)*(1 - A/K_R_t) - (sigma + mu_A) * A
            dSv_dt = sigma * A - beta_hv_t * Sv * (Ih / Nh) - mu_v_t * Sv
            dIv_dt = beta_hv_t * Sv * (Ih / Nh) - mu_v_t * Iv
            dSh_dt = Lambda_h - beta_vh_t * Sh * (Iv / Nh) - mu_h * Sh + omega * Rh
            dIh_dt = beta_vh_t * Sh * (Iv / Nh) - (r + mu_h + delta) * Ih
            dRh_dt = r * Ih - mu_h * Rh - omega * Rh
            return [dA_dt, dSv_dt, dIv_dt, dSh_dt, dIh_dt, dRh_dt]

        A0 = df_iklim['K_R'].values[0]
        Sv0 = (sigma * A0) / df_iklim['MU_v'].values[0]
        y0 = [A0, Sv0, 50, Nh - 100, 100, 0] 
        
        hasil_ode = odeint(sirlm_model, y0, indeks_waktu)
        A_out, Sv_out, Iv_out, Sh_out, Ih_out, Rh_out = hasil_ode.T

        m = np.maximum(Sv_out / Nh, 1e-9)
        a_t_array = np.maximum(df_iklim['BETA_hv'].values / p_hv, 1e-9) 
        R0_t = (m * (a_t_array**2) * (p_hv * p_vh)) / (r * df_iklim['MU_v'].values)
        df_iklim['R0_t'] = R0_t
        df_iklim['Simulasi_Ih'] = Ih_out
        df_simulasi_mingguan = df_iklim['Simulasi_Ih'].resample('W-MON').mean()

        # --- D. PERHITUNGAN CROSS CORRELATION ---
        best_lag = "N/A"; max_corr = 0
        if not is_mandiri:
            df_r0_mingguan = df_iklim['R0_t'].resample('W-MON').mean().reset_index()
            df_r0_mingguan.rename(columns={'Date': 'Tanggal_Minggu'}, inplace=True) 
            df_bpjs_mingguan_reset = df_bpjs_mingguan.reset_index()
            df_merged = pd.merge(df_r0_mingguan, df_bpjs_mingguan_reset, on='Tanggal_Minggu', how='inner')
            
            if len(df_merged) > 30: 
                r0_series = (df_merged['R0_t'] - df_merged['R0_t'].mean()) / df_merged['R0_t'].std()
                kasus_series = (df_merged['Insiden_Kasus'] - df_merged['Insiden_Kasus'].mean()) / df_merged['Insiden_Kasus'].std()
                lags = np.arange(-24, 25) 
                cross_corr = [r0_series.corr(kasus_series.shift(lag)) for lag in lags]
                best_lag = lags[np.nanargmax(cross_corr)]
                max_corr = np.nanmax(cross_corr)

        # ==========================================
        # 5. VISUALISASI HASIL & TABS KONTROL
        # ==========================================
        st.markdown("---")
        st.markdown("### Ringkasan Hasil Simulasi")
        
        # Penyesuaian ke 3 Kolom Saja
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Puncak Kasus Infeksi (Ih)", f"{int(max(Ih_out))} Jiwa")
        col_kpi2.metric("Rata-rata R0(t)", f"{R0_t.mean():.2f}")
        col_kpi3.metric("Maksimum R0(t)", f"{R0_t.max():.2f}")

        if is_mandiri:
            tab1, tab2, tab3 = st.tabs(["Hasil Simulasi Infeksi", "Manusia & Nyamuk", "Fase Jentik & Potensi Wabah (R0)"])
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["BPJS vs Simulasi", "Manusia & Nyamuk", "Fase Jentik", "Wabah (R0) & Korelasi"])

        with tab1:
            st.subheader("Kurva Kasus Infeksius Manusia Hasil Simulasi")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_simulasi_mingguan.index, y=df_simulasi_mingguan.values, name="Simulasi (I_h)", line=dict(color='red', width=2)))
            if not is_mandiri:
                fig_comp.add_trace(go.Scatter(x=df_bpjs_mingguan.index, y=df_bpjs_mingguan['Insiden_Kasus'], name="Aktual BPJS", line=dict(color='teal', width=2.5)))
                st.info("Interpretasi: Membandingkan akurasi kurva matematika (merah) dalam meniru lonjakan kasus aktual BPJS (teal).")
            fig_comp.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Kasus", hovermode="x unified")
            st.plotly_chart(fig_comp, width='stretch')

        with tab2:
            st.subheader("Dinamika Populasi Manusia (SIR)")
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Sh_out, name="Rentan (S_h)", line=dict(color='#2980b9')))
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Ih_out, name="Terinfeksi (I_h)", line=dict(color='#e74c3c', width=2.5)))
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Rh_out, name="Sembuh (R_h)", line=dict(color='#2ecc71')))
            fig_h.update_layout(xaxis_title="Tanggal", yaxis_title="Populasi", hovermode="x unified")
            st.plotly_chart(fig_h, width='stretch')

            st.subheader("Dinamika Populasi Nyamuk Dewasa")
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(x=tanggal_iklim, y=Sv_out, name="Nyamuk Rentan (S_v)", line=dict(color='#f39c12')))
            fig_v.add_trace(go.Scatter(x=tanggal_iklim, y=Iv_out, name="Nyamuk Infeksius (I_v)", line=dict(color='#d35400', width=2.5)))
            fig_v.update_layout(xaxis_title="Tanggal", yaxis_title="Populasi Nyamuk", hovermode="x unified")
            st.plotly_chart(fig_v, width='stretch')

        with tab3:
            st.subheader("Dinamika Fase Akuatik / Jentik Nyamuk")
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=tanggal_iklim, y=A_out, name="Fase Akuatik (A)", line=dict(color='#8e44ad', width=2.5)))
            fig_a.add_trace(go.Scatter(x=tanggal_iklim, y=df_iklim['K_R'].values, name="Daya Tampung Genangan (K_R)", line=dict(color='#3498db', dash='dot')))
            fig_a.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Individu", hovermode="x unified")
            st.plotly_chart(fig_a, width='stretch')

            if is_mandiri:
                st.markdown("---")
                st.subheader("Angka Reproduksi Dasar Berbasis Iklim (R0(t))")
                fig_r0 = go.Figure()
                fig_r0.add_trace(go.Scatter(x=tanggal_iklim, y=R0_t, name="Nilai R0(t)", line=dict(color='#9b59b6', width=2)))
                fig_r0.add_shape(type="line", x0=tanggal_iklim.min(), y0=1.0, x1=tanggal_iklim.max(), y1=1.0, line=dict(color="red", dash="dash", width=2))
                fig_r0.update_layout(xaxis_title="Tanggal", yaxis_title="Nilai R0", hovermode="x unified")
                st.plotly_chart(fig_r0, width='stretch')
                
                st.info(
                    "Cara Membaca Grafik R0(t) (Angka Reproduksi Dasar Dinamis):\n"
                    "* Grafik ini mengevaluasi potensi wabah yang berfluktuasi berdasarkan iklim harian.\n"
                    "* Garis Merah Putus-putus (R0 = 1): Adalah batas ambang penyebaran penyakit.\n"
                    "* Jika Kurva R0 > 1 (Di atas garis merah): Menandakan Transmisi Persisten. Cuaca sangat ideal bagi nyamuk untuk berkembang biak, sehingga satu orang yang terinfeksi berpotensi menularkan malaria ke lebih dari satu orang lainnya (risiko letusan kasus).\n"
                    "* Jika Kurva R0 < 1 (Di bawah garis merah): Penularan cenderung sporadis atau terhenti. Kondisi iklim ekstrem (terlalu dingin/kurang hujan) bertindak sebagai pembatas alami yang menghambat reproduksi vektor nyamuk."
                )

        if not is_mandiri:
            with tab4:
                st.subheader("Angka Reproduksi Dasar Berbasis Iklim (R0(t))")
                fig_r0 = go.Figure()
                fig_r0.add_trace(go.Scatter(x=tanggal_iklim, y=R0_t, name="Nilai R0(t)", line=dict(color='#9b59b6', width=2)))
                fig_r0.add_shape(type="line", x0=tanggal_iklim.min(), y0=1.0, x1=tanggal_iklim.max(), y1=1.0, line=dict(color="red", dash="dash", width=2))
                fig_r0.update_layout(xaxis_title="Tanggal", yaxis_title="Nilai R0", hovermode="x unified")
                st.plotly_chart(fig_r0, width='stretch')
                
                st.info(
                    "Cara Membaca Grafik R0(t) (Angka Reproduksi Dasar Dinamis):\n"
                    "* Grafik ini mengevaluasi potensi wabah yang berfluktuasi berdasarkan iklim harian.\n"
                    "* Garis Merah Putus-putus (R0 = 1): Adalah batas ambang penyebaran penyakit.\n"
                    "* Jika Kurva R0 > 1 (Di atas garis merah): Menandakan Transmisi Persisten. Cuaca sangat ideal bagi nyamuk untuk berkembang biak, sehingga satu orang yang terinfeksi berpotensi menularkan malaria ke lebih dari satu orang lainnya (risiko letusan kasus).\n"
                    "* Jika Kurva R0 < 1 (Di bawah garis merah): Penularan cenderung sporadis atau terhenti. Kondisi iklim ekstrem (terlalu dingin/kurang hujan) bertindak sebagai pembatas alami yang menghambat reproduksi vektor nyamuk."
                )

                st.markdown("---")
                st.subheader("Cross-Correlation: Jeda Iklim vs Insiden Aktual")
                if len(df_merged) > 30: 
                    fig_cc = go.Figure()
                    fig_cc.add_trace(go.Bar(x=lags, y=cross_corr, marker_color='indigo', name='Korelasi'))
                    fig_cc.update_layout(xaxis_title="Lag (Minggu)", yaxis_title="Korelasi Pearson")
                    st.plotly_chart(fig_cc, width='stretch')
                    
                    if best_lag < 0:
                        st.success(f"Kesimpulan: Iklim memicu lonjakan insiden dengan jeda (Lag) {-best_lag} minggu (Korelasi maks: {max_corr:.3f}).")
                    elif best_lag > 0:
                        st.info(f"Kesimpulan: Insiden mendahului iklim dengan jeda {best_lag} minggu (Korelasi maks: {max_corr:.3f}).")
                else:
                    st.warning("Waktu antara data NASA dan BPJS kurang dari 30 minggu, Cross-Correlation tidak dapat dihitung.")

        st.markdown("---")
        df_hasil_download = pd.DataFrame({
            'Tanggal': tanggal_iklim, 'S_Manusia': Sh_out, 'I_Manusia': Ih_out,
            'S_Nyamuk_Dewasa': Sv_out, 'I_Nyamuk_Dewasa': Iv_out, 'Jentik': A_out, 'Nilai_R0': R0_t
        })
        st.download_button(
            label="Unduh Data Hasil Simulasi (CSV)",
            data=df_hasil_download.to_csv(index=False).encode('utf-8'),
            file_name=f"Simulasi_{pilihan_wilayah}.csv", mime="text/csv",
        )

except Exception as e:
    st.error(f"Terjadi kesalahan di sistem: {e}")