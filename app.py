import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# ==========================================
# 1. KONFIGURASI HALAMAN & TEORI
# ==========================================
st.set_page_config(page_title="Simulasi Epidemiologi Malaria", layout="wide", page_icon="🦟")

st.title("📊 Sistem Simulasi Malaria Universal (Model SIR-LM)")
st.markdown("Aplikasi interaktif pemodelan dinamika malaria berbasis iklim untuk kabupaten/kota manapun.")

# [REKOMENDASI 2] Penjelasan Teori & Rumus
with st.expander("📖 Pelajari Model Matematika SIR-LM yang Digunakan"):
    st.markdown("""
    Sistem ini menggunakan penggabungan model **SIR (Susceptible-Infected-Recovered)** untuk dinamika manusia, dan model **LM (Life-cycle Malaria)** untuk vektor nyamuk yang mencakup fase akuatik (jentik).
    """)
    st.markdown("**1. Model Dinamika Manusia (SIR)**")
    st.latex(r''' \frac{dS_h}{dt} = \Lambda_h - \beta_{vh}(t) S_h \frac{I_v}{N_h} - \mu_h S_h + \omega R_h ''')
    st.markdown("**2. Dinamika Nyamuk Dewasa & Fase Akuatik (LM)**")
    st.latex(r''' \frac{dA}{dt} = \phi (S_v + I_v)\left(1 - \frac{A}{K_R(t)}\right) - (\sigma + \mu_A)A ''')
    st.info("💡 **K_R(t)** adalah kapasitas daya tampung lingkungan berbasis curah hujan (kumulatif 14 hari).")

# ==========================================
# 2. SIDEBAR: PENGATURAN KELOMPOK
# ==========================================
st.sidebar.header("⚙️ Pengaturan Simulasi")

# [REKOMENDASI 3] Sidebar yang dikelompokkan
with st.sidebar.expander("🌍 Parameter Wilayah & Demografi", expanded=True):
    nama_wilayah = st.text_input("📍 Nama Wilayah:", value="Kota Bandung")
    Nh_input = st.number_input("👥 Total Populasi Penduduk:", value=1000000, step=10000)

with st.sidebar.expander("🦟 Parameter Vektor & Iklim", expanded=True):
    k_base = st.number_input("K_min (Daya Tampung Dasar)", min_value=1000, value=15000, step=500)
    alpha = st.number_input("Theta (Faktor Konversi Hujan)", min_value=10.0, value=350.0, step=10.0)
    gap_hari = st.number_input("Deduplikasi Episode BPJS (Hari)", min_value=7, value=30, step=1)

# ==========================================
# 3. KOTAK UPLOAD KEDUA FILE UTAMA
# ==========================================
st.markdown("### 📥 Unggah Dataset Wilayah")
col_up1, col_up2 = st.columns(2)
with col_up1:
    file_iklim = st.file_uploader("1. Unggah Data Iklim NASA (Format CSV)", type=["csv"])
with col_up2:
    file_bpjs = st.file_uploader("2. Unggah Data Kunjungan BPJS (Format Excel/CSV)", type=["xlsx", "xls", "csv"])

if file_iklim is None or file_bpjs is None:
    st.info("💡 Menunggu data... Silakan unggah **KEDUA** file pada kotak di atas untuk memulai simulasi sistem.")
    st.stop()

# ==========================================
# 4. PROSES DATA & PEMODELAN MATEMATIKA
# ==========================================
try:
    with st.spinner(f"Memproses data dan memodelkan dinamika untuk {nama_wilayah}..."):
        
        # --- A. PROSES DATA BPJS ---
        if file_bpjs.name.endswith('.csv'):
            df_bpjs = pd.read_csv(file_bpjs)
        else:
            df_bpjs = pd.read_excel(file_bpjs)
            
        df_bpjs['Tgl_Start'] = pd.to_datetime(df_bpjs['Tgl_Start'])
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
        try:
            df_iklim = pd.read_csv(file_iklim, skiprows=12)
            df_iklim.replace(-999.0, np.nan, inplace=True)
            if 'T2M' not in df_iklim.columns:
                file_iklim.seek(0)
                df_iklim = pd.read_csv(file_iklim)
        except:
            file_iklim.seek(0)
            df_iklim = pd.read_csv(file_iklim)
            
        df_iklim['Date'] = pd.to_datetime(df_iklim['YEAR'] * 1000 + df_iklim['DOY'], format='%Y%j')
        df_iklim.set_index('Date', inplace=True)
        
        df_iklim['T2M'] = df_iklim['T2M'].interpolate(method='linear').fillna(24.0)
        df_iklim['PRECTOTCORR'] = df_iklim['PRECTOTCORR'].fillna(0)

        threshold_hujan = 150.0
        df_iklim['PRECTOTCORR_ROLL14'] = df_iklim['PRECTOTCORR'].clip(upper=threshold_hujan).rolling(window=14).sum().shift(1)
        df_iklim.dropna(subset=['PRECTOTCORR_ROLL14'], inplace=True)
        indeks_waktu = np.arange(len(df_iklim))
        tanggal_iklim = df_iklim.index

        # [REKOMENDASI 4] Pratinjau Data (Data Preview)
        with st.expander("👀 Lihat Pratinjau Data yang Berhasil Diproses"):
            col_prev1, col_prev2 = st.columns(2)
            with col_prev1:
                st.markdown("**Data Iklim Harian**")
                st.dataframe(df_iklim[['T2M', 'PRECTOTCORR']].head())
            with col_prev2:
                st.markdown("**Data Agregat Kasus Mingguan (BPJS)**")
                st.dataframe(df_bpjs_mingguan.head())

        # --- C. SISTEM ODE (SIR-LM MODEL) ---
        c_briere = 0.000116
        T_min_b = 15.64 
        T_max_b = 34.92 
        p_hv = 0.75 
        p_vh = 0.5 
        
        def briere_function(T):
            return np.where((T > T_min_b) & (T < T_max_b), c_briere * T * (T - T_min_b) * np.sqrt(T_max_b - T), 0)

        b_T = briere_function(df_iklim['T2M'])
        df_iklim['BETA_hv'] = b_T * p_hv * 3.5
        df_iklim['BETA_vh'] = b_T * p_vh * 3.5

        def mortality_function(T):
            mu = 0.002 * (T - 35.0)**2 + 0.05
            return np.minimum(mu, 1.0)
            
        df_iklim['MU_v'] = mortality_function(df_iklim['T2M'])
        df_iklim['K_R'] = k_base + alpha * df_iklim['PRECTOTCORR_ROLL14']

        K_func = interp1d(indeks_waktu, df_iklim['K_R'].values, kind='linear', fill_value="extrapolate")
        Beta_hv_func = interp1d(indeks_waktu, df_iklim['BETA_hv'].values, kind='linear', fill_value="extrapolate")
        Beta_vh_func = interp1d(indeks_waktu, df_iklim['BETA_vh'].values, kind='linear', fill_value="extrapolate")
        MU_v_func = interp1d(indeks_waktu, df_iklim['MU_v'].values, kind='linear', fill_value="extrapolate")

        Nh = Nh_input
        mu_h = 1 / (66.44 * 365)
        Lambda_h = (0.0223 * Nh) / 365
        omega = 1 / 120
        r = 1 / 14 
        delta = 0.01 
        phi = 3.8 
        sigma = 0.07 
        mu_A = 0.32 

        def sirlm_model(y, t):
            A, Sv, Iv, Sh, Ih, Rh = y
            
            K_R_t = max(K_func(t), 1.0)
            beta_hv_t = max(Beta_hv_func(t), 0.0)
            beta_vh_t = max(Beta_vh_func(t), 0.0)
            mu_v_t = max(MU_v_func(t), 0.0)
            
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
        b_c = p_hv * p_vh
        R0_t = (m * (a_t_array**2) * b_c) / (r * df_iklim['MU_v'].values)
        df_iklim['R0_t'] = R0_t

        df_iklim['Simulasi_Ih'] = Ih_out
        df_simulasi_mingguan = df_iklim['Simulasi_Ih'].resample('W-MON').mean()

        # --- D. PERHITUNGAN CROSS CORRELATION (Untuk dimasukkan ke KPI Metriks) ---
        df_r0_mingguan = df_iklim['R0_t'].resample('W-MON').mean().reset_index()
        df_r0_mingguan.rename(columns={'Date': 'Tanggal_Minggu'}, inplace=True) # FIX ERROR TANGGAL
        df_bpjs_mingguan_reset = df_bpjs_mingguan.reset_index()
        df_merged = pd.merge(df_r0_mingguan, df_bpjs_mingguan_reset, on='Tanggal_Minggu', how='inner')
        
        best_lag = None
        max_corr = None
        if len(df_merged) > 30: 
            r0_series = (df_merged['R0_t'] - df_merged['R0_t'].mean()) / df_merged['R0_t'].std()
            kasus_series = (df_merged['Insiden_Kasus'] - df_merged['Insiden_Kasus'].mean()) / df_merged['Insiden_Kasus'].std()
            lags = np.arange(-24, 25) 
            cross_corr = [r0_series.corr(kasus_series.shift(lag)) for lag in lags]
            best_lag = lags[np.nanargmax(cross_corr)]
            max_corr = np.nanmax(cross_corr)

        st.success(f"✅ Seluruh model untuk {nama_wilayah} berhasil dikalkulasi!")

        # ==========================================
        # 5. VISUALISASI HASIL & KPI METRICS
        # ==========================================
        st.markdown("---")
        
        # [REKOMENDASI 1] Panel Angka Ringkasan (KPI Metrics)
        st.markdown(f"### 📌 Ringkasan Hasil Simulasi: {nama_wilayah}")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Puncak Kasus Infeksi", f"{int(max(Ih_out))} Jiwa", "Berdasarkan Simulasi")
        col_kpi2.metric("Rata-rata R0(t)", f"{R0_t.mean():.2f}")
        col_kpi3.metric("Maksimum R0(t)", f"{R0_t.max():.2f}", "Potensi Wabah", delta_color="inverse")
        col_kpi4.metric("Korelasi Terbaik (Lag)", f"{best_lag} Minggu" if best_lag is not None else "N/A", "Iklim vs Aktual")

        # [REKOMENDASI 5] Tombol Download CSV
        df_hasil_download = pd.DataFrame({
            'Tanggal': tanggal_iklim,
            'S_Manusia': Sh_out,
            'I_Manusia': Ih_out,
            'R_Manusia': Rh_out,
            'S_Nyamuk_Dewasa': Sv_out,
            'I_Nyamuk_Dewasa': Iv_out,
            'Fase_Akuatik_Jentik': A_out,
            'Nilai_R0': R0_t
        })
        csv_download = df_hasil_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Data Hasil Simulasi (CSV)",
            data=csv_download,
            file_name=f"Hasil_Simulasi_{nama_wilayah}.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📉 BPJS vs Simulasi", "👥 Manusia & Nyamuk", "💧 Fase Jentik", "⚠️ Wabah & Korelasi"])
        
        with tab1:
            st.subheader(f"Perbandingan Kasus Aktual vs Simulasi Model ({nama_wilayah})")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_simulasi_mingguan.index, y=df_simulasi_mingguan.values, name="Simulasi Terinfeksi (I_h)", line=dict(color='red', width=2)))
            fig_comp.add_trace(go.Scatter(x=df_bpjs_mingguan.index, y=df_bpjs_mingguan['Insiden_Kasus'], name="Aktual BPJS", line=dict(color='teal', width=2.5)))
            fig_comp.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Kasus", hovermode="x unified")
            st.plotly_chart(fig_comp, width='stretch')
            # [REKOMENDASI 6] Penjelasan
            st.info("💡 **Interpretasi:** Grafik ini menunjukkan seberapa akurat model matematika (merah) meniru perilaku kasus asli yang tercatat di fasilitas kesehatan (teal).")
            
        with tab2:
            st.subheader("Dinamika Populasi Manusia (SIR)")
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Sh_out, name="Rentan (S_h)", line=dict(color='#2980b9')))
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Ih_out, name="Terinfeksi (I_h)", line=dict(color='#e74c3c', width=2.5)))
            fig_h.add_trace(go.Scatter(x=tanggal_iklim, y=Rh_out, name="Sembuh (R_h)", line=dict(color='#2ecc71')))
            fig_h.update_layout(xaxis_title="Tanggal", yaxis_title="Populasi Manusia", hovermode="x unified")
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
            st.info("💡 **Interpretasi:** Daya tampung lingkungan membatasi populasi jentik. Hujan terus-menerus akan meningkatkan nilai batas K_R (garis biru).")

        with tab4:
            st.subheader("Angka Reproduksi Dasar Berbasis Iklim ($R_0(t)$)")
            fig_r0 = go.Figure()
            fig_r0.add_trace(go.Scatter(x=tanggal_iklim, y=R0_t, name="Nilai R0(t)", line=dict(color='#9b59b6', width=2)))
            fig_r0.add_shape(type="line", x0=tanggal_iklim.min(), y0=1.0, x1=tanggal_iklim.max(), y1=1.0, line=dict(color="red", dash="dash", width=2))
            fig_r0.update_layout(xaxis_title="Tanggal", yaxis_title="Nilai R0", hovermode="x unified")
            st.plotly_chart(fig_r0, width='stretch')
            if R0_t.max() > 1.0:
                st.warning("⚠️ **Perhatian:** Ditemukan periode waktu di mana nilai $R_0 > 1$, yang berarti secara teori berpotensi kuat memicu terjadinya letusan wabah malaria baru.")

            # --- ANALISIS CROSS CORRELATION ---
            st.subheader(f"Cross-Correlation: $R_0(t)$ Iklim vs Insiden Aktual ({nama_wilayah})")
            if len(df_merged) > 30: 
                fig_cc = go.Figure()
                fig_cc.add_trace(go.Bar(x=lags, y=cross_corr, marker_color='indigo', name='Korelasi'))
                fig_cc.update_layout(
                    xaxis_title="Lag (Minggu)", yaxis_title="Korelasi Pearson", 
                    title="Seberapa jauh iklim mendahului lonjakan kasus malaria?"
                )
                st.plotly_chart(fig_cc, width='stretch')
                
                if best_lag < 0:
                    st.success(f"💡 **Kesimpulan Aktif:** $R_0(t)$ (iklim) memimpin (mendahului) lonjakan insiden kasus asli dengan jeda **{-best_lag} minggu** (Korelasi maksimal: {max_corr:.3f}).")
                elif best_lag > 0:
                    st.info(f"💡 **Kesimpulan:** Insiden kasus mendahului $R_0(t)$ dengan jeda **{best_lag} minggu** (Korelasi maksimal: {max_corr:.3f}).")
                else:
                    st.info(f"💡 **Kesimpulan:** $R_0(t)$ dan kasus saling berkaitan di minggu yang sama secara langsung (Korelasi maksimal: {max_corr:.3f}).")
            else:
                st.warning("⚠️ Waktu antara data NASA dan BPJS kurang dari 30 minggu, Cross-Correlation tidak dapat dihitung.")
            
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan dalam pemrosesan data: {e}")