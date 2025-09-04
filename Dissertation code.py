#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========
# Imports    
# ==========
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
from scipy.signal import butter, filtfilt, welch, detrend, correlate, correlation_lags
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker as mtick
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# In[2]:


# ========
# Setup
# ========
# plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# saving figures 
fig_dir = Path('figures')
fig_dir.mkdir(parents=True, exist_ok=True)

# parameters for reproducibility and analysis
random_state = 1       # seed
n_days = 14            # number of days analysed
fs = 100.0             # sampling rate in Hz
win_s = 5              # seconds per waveform window
wlen = int(fs * win_s) # samples per waveform window

# high-pass filter settings
cutoff, order = 1.0, 4
b, a = butter(order, cutoff / (fs/2), btype='highpass')

# k-means settings 
n_init = 100               # number of initialisations
kmeans_tol = 1e-3          # per-iteration convergence threshold
max_kmeans_iter = 1000     # cap
kmeans_algorithm = 'elkan' # faster for euclidean distances 


# In[3]:


# ==================================
# Load HDF5 file and extract data 
# ==================================
# path
h5_path = Path('/home3/prwx27/1. MH-Dissertation/dataset/ev0000593283.h5')
if not h5_path.is_file():
    raise FileNotFoundError(f'HDF5 not found: {h5_path}')

# stations analysed
stations = ['IMRH', 'UWEH']
station_a = stations[0]
station_b = stations[1]

# extract file metadata
with h5py.File(h5_path, 'r') as f_meta:
    if any(st not in f_meta for st in stations):
        missing = [st for st in stations if st not in f_meta]
        raise KeyError(f'Missing station(s): {missing}')
    
    # file start timestamp (UTC)
    raw_ts = f_meta.attrs.get('time')
    if isinstance(raw_ts, (bytes, bytearray)):
        raw_ts = raw_ts.decode('utf-8')
    wave_file_start = pd.to_datetime(raw_ts, utc=True)
    print(f'File start time: {wave_file_start}')

    # usable duration across stations (min length)
    station_lengths = {st: f_meta[st].shape[1] for st in stations}
    max_samples = min(station_lengths.values())
    duration_days = max_samples / (fs * 3600 * 24)
    print(f'Usable file duration: {duration_days:.2f} days\n')

    # station coordinates for distance-based quake filtering later
    station_coords = {
        st: (float(f_meta[st].attrs['lat']), float(f_meta[st].attrs['lon']))
        for st in stations
    }


# In[4]:


# ==================================
# Build map of station locations
# ==================================
# to show on introduction section

# map display names
station_display = {
    'IMRH': 'SAGH02',
    'UWEH': 'KMMH13'
}

# build GeoDataFrame with display names
gdf = gpd.GeoDataFrame(
    {'name': [station_display[s] for s in stations],
     'geometry': [Point(station_coords[s][1], station_coords[s][0]) for s in stations]},
    crs='EPSG:4326'
).to_crs(3857)

# plot overview map
fig, ax = plt.subplots(figsize=(6,6))
gdf.plot(ax=ax, color='blue', markersize=90, edgecolor='white', linewidth=0.6, zorder=3)

# zoom
buffer_km = 80
xmin, ymin, xmax, ymax = gdf.total_bounds
pad = buffer_km * 1000
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# english labels via CartoDB Positron
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf.crs)

# labels
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['name']):
    ax.text(x, y, label, fontsize=10, weight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7), zorder=4)

ax.set_axis_off()
fig.tight_layout()
fig.savefig(fig_dir / 'stations_map.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)

# detailed map per station
detail_buffer_km = 5
for st in stations:
    display_name = station_display[st]
    row = gdf[gdf['name'] == display_name].iloc[0]
    x, y = row.geometry.x, row.geometry.y

    fig, ax = plt.subplots(figsize=(6, 6))
    gdf[gdf['name'] == display_name].plot(ax=ax, color='blue', markersize=90, edgecolor='white', linewidth=0.6, zorder=3)

    pad = detail_buffer_km * 1000
    ax.set_xlim(x - pad, x + pad)
    ax.set_ylim(y - pad, y + pad)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=gdf.crs)

    ax.text(x, y, display_name, fontsize=10, weight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7), zorder=4)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fig_dir / f'station_{display_name}_detail.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# In[5]:


# ============================================================
# Define the time period used as input to the cluster model
# ============================================================
# two continuous weeks: 2016-04-18 00:00 to 2016-05-02 00:00 (UTC)                         
start_date = pd.Timestamp('2016-04-18 00:00:00', tz='UTC')
day_samples = int(24 * 3600 * fs)
ns_per_sample = int(1e9 / fs) # exact nanoseconds per sample to avoid float drift

# compute daily start/end sample indices relative to file start
sample_periods = {}
for day in pd.date_range(start_date, periods=n_days, freq='D', tz='UTC'):
    label = day.strftime('%Y-%m-%d')
    delta_ns = day.value - wave_file_start.value
    i0 = int(round(delta_ns * fs / 1e9))
    i1 = i0 + day_samples
    sample_periods[label] = (i0, i1)
    
    # reconstruct timestamp from samples check
    ts0 = wave_file_start + pd.Timedelta(nanoseconds=i0 * ns_per_sample)
    print(f'{label}: samples {i0} to {i1}  |  start timestamp: {ts0}')


# In[6]:


# ======================
# Feature engineering  
# ======================
# function to compute an array of features per window
def extract_features(x: np.ndarray, fs: float) -> np.ndarray:
    # time domain energy proxy
    dt = 1 / fs
    dt_sq_trace = np.sum(x**2) * dt

    # frequency domain features
    freqs, psd = welch(x, fs=fs, nperseg=len(x))
    df = freqs[1] - freqs[0]
    amp = np.sqrt(psd * df)

    max_amp = float(amp.max()) # peak amplitude spectral density bin
    idx_max = int(np.argmax(amp))
    freq_max_amp = float(freqs[idx_max]) # frequency of that peak bin

    psd_sum = np.sum(psd)
    center_freq = float(np.sum(freqs * psd) / psd_sum) # psd-weighted spectral centroid
    bandwidth = float(np.sqrt(np.sum((freqs - center_freq) ** 2 * psd) / psd_sum)) # psd-weighted spread about centroid

    # time domain zero-crossing rate
    zcr = float(np.sum(np.signbit(x[:-1]) != np.signbit(x[1:])) / (len(x) / fs)) # number of sign changes per second

    # frequency domain spectral sharpness proxy
    omega = 2 * np.pi * freqs
    psd2 = psd**2
    num2 = np.sum(omega**4 * psd2)
    den2 = np.sum(omega**2 * psd2)
    peak_rate = float(np.sqrt(num2 / den2) if den2 > 0 else 0.0)

    return np.array([dt_sq_trace, max_amp, freq_max_amp, center_freq, bandwidth, zcr, peak_rate], dtype=float)

feature_names = ['dt_sq_trace', 'max_amp', 'freq_max_amp', 'center_freq', 'bandwidth', 'zcr', 'peak_rate']


# In[7]:


# ======================== STATION A ===========================

# ============================================
# Data preprocessing and feature extraction 
# ============================================
# containers
meta_records = []
features_list = []

with h5py.File(h5_path, 'r') as f:
    for period, (i0, i1) in sample_periods.items():
        nwin = (i1 - i0) // wlen
        
        # load full day slices and perform computations
        x_raw = f[station_a][0, i0:i1]
        x_dt = detrend(x_raw, type='linear') # apply detrend 
        x_f = filtfilt(b, a, x_dt) # apply high-pass filter 

        # slide over 5 s windows 
        for wi in range(nwin):
            s0 = i0 + wi * wlen
            x_seg = x_f[wi*wlen : wi*wlen + wlen]
                
            # extract features 
            feats = extract_features(x_seg, fs)
            features_list.append(feats)

            # store metadata  
            ts = wave_file_start + pd.Timedelta(seconds=s0/fs)
            meta_records.append({
                'period': period,
                'station': station_a,
                'window_index': wi,
                'abs_sample_idx': s0,
                'timestamp': ts
            })

# store all metadata and features 
meta_df = pd.DataFrame(meta_records)
features_flat = np.vstack(features_list)

print(meta_df.head(), '\n')
print(meta_df['station'].value_counts())
print('features_flat shape:', features_flat.shape)


# In[8]:


# ==========================
# Standardisation and pca
# ==========================
# standardise features to zero mean and unit variance
features_scaled = StandardScaler().fit_transform(features_flat)

# fit full pca to inspect explained variance 
pca_full = PCA(random_state=random_state, whiten=False)
Xp_full = pca_full.fit_transform(features_scaled)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
print('Cumulative explained variance:', cumulative)

# compute covariance matrix 
cov_matrix = np.cov(Xp_full, rowvar=False)
print('Covariance matrix (rounded):\n', np.round(cov_matrix, 3))

# scree plot 
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(1, len(explained) + 1)
ax.plot(xs, explained, 'ok--')
ax.set_xlabel('Principal Component Number')
ax.set_ylabel('Explained Variance')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.grid(True)
fig.tight_layout()
fig.savefig(fig_dir / 'scree_plot.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)

# keep number of pcs explaining >90% of variance for clustering
target_variance = 0.90
n_components = int(np.searchsorted(cumulative, target_variance) + 1)
print(f'Selected {n_components} PCs')

pca_n = PCA(n_components=n_components, random_state=random_state, whiten=False)
Xp = pca_n.fit_transform(features_scaled)
print('Xp shape:', Xp.shape)


# In[9]:


# =========================================
# Optimal k selection for k-means model
# =========================================
# elbow method over a wide k range for context 
cluster_range = list(range(2, 21))
wss = []
for k in cluster_range:
    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=n_init,
        tol=kmeans_tol,
        max_iter=max_kmeans_iter,
        algorithm=kmeans_algorithm,
        random_state=random_state,
    )    
    km.fit(Xp)
    wss.append(km.inertia_)

# plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cluster_range, wss, 'ok--')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Within-Cluster Sum of Squares')
ax.set_xticks(cluster_range)

fig.tight_layout()
fig.savefig(fig_dir / 'elbow_plot.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)


# In[10]:


# silhouette analysis on k's at the curve in the elbow plot
base_cmap = plt.get_cmap('tab10') # colour palette 
sil_cluster_range = list(range(4, 8))

# containers 
all_km, all_labels, all_sil = {}, {}, {}
sil_avgs = []

# model 
# silhouette analysis adapted from: The scikit-learn developers
# source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
for k in sil_cluster_range:
    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=n_init,
        tol=kmeans_tol,
        max_iter=max_kmeans_iter,
        algorithm=kmeans_algorithm,
        random_state=random_state,
    )    
    labels = km.fit_predict(Xp)
    sil_vals = silhouette_samples(Xp, labels)
    sil_avg = silhouette_score(Xp, labels)

    # store for downstream use
    all_km[k], all_labels[k], all_sil[k] = km, labels, sil_vals
    sil_avgs.append(sil_avg)

    # determine cluster sizes and sort clusters by size for consistency across plots
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    sorted_clusters = sorted(range(k), key=lambda i: cluster_sizes[i], reverse=True)
    new_label_map = {orig: new for new, orig in enumerate(sorted_clusters)}

    remapped_labels = np.array([new_label_map[orig] for orig in labels])
    remapped_centers = np.array([km.cluster_centers_[orig] for orig in sorted_clusters])

    # silhouette bars + pc scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # silhouette subplot 
    ax1.set_title(f'Silhouette Analysis (k = {k})')
    ax1.set_xlim([-0.2, 1])
    ax1.set_xticks(np.arange(-0.2, 1.05, 0.1))
    ax1.set_ylim([0, len(Xp) + (k + 1) * 10])
    y_lower = 10
    for new_label, orig in enumerate(sorted_clusters):
        ith_vals = np.sort(sil_vals[labels == orig])
        y_upper = y_lower + len(ith_vals)
        color = base_cmap(new_label % base_cmap.N)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=color, edgecolor=color, alpha=0.7)
        x_text = -0.05 if new_label % 2 == 0 else 0.05
        ax1.text(x_text, y_lower + 0.5 * len(ith_vals), str(new_label + 1))
        y_lower = y_upper + 10
    ax1.axvline(sil_avg, color='red', linestyle='--')
    ax1.set_xlabel('Silhouette Score')
    ax1.set_ylabel('Number of Samples')

    # pc1 vs pc2 scatter with centroids
    scatter_colors = [base_cmap(l % base_cmap.N) for l in remapped_labels]
    ax2.scatter(Xp[:, 0], Xp[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=scatter_colors)
    ax2.scatter(remapped_centers[:, 0], remapped_centers[:, 1], marker='o', c='white', s=200, edgecolor='k')
    for idx, c in enumerate(remapped_centers):
        ax2.scatter(c[0], c[1], marker=f'${idx+1}$', edgecolor='k')
    ax2.set_title('Cluster Visualisation (PC1 vs PC2)')
    fig.tight_layout()
    fig.savefig(fig_dir / f'silhouette_k{k}.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f'k = {k} has avg silhouette = {sil_avg:.4f}')
    
# summary plot of average silhouette across chosen k's
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sil_cluster_range, sil_avgs, 'ok--')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Average Silhouette Score')
ax.set_xticks(sil_cluster_range)
ax.grid(True)

fig.tight_layout()
fig.savefig(fig_dir / 'silhouette_vs_nclusters.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)


# In[11]:


# ===============================================================
# Finalise k, relabel by size, and join labels back to windows 
# ===============================================================
# keep model with best k score
best_idx = int(np.argmax(sil_avgs))
best_k = sil_cluster_range[best_idx]

labels_orig = all_labels[best_k]
sil_best = all_sil[best_k]
km_best = all_km[best_k]

# map original labels to new labels where 0 is the largest cluster
cluster_sizes = pd.Series(labels_orig).value_counts().to_dict()
sorted_clusters_orig = sorted(cluster_sizes, key=lambda x: cluster_sizes[x], reverse=True)
new_label_map = {orig: new for new, orig in enumerate(sorted_clusters_orig)}
labels_best = np.array([new_label_map[l] for l in labels_orig])

# shift labels by 1
labels_best = labels_best + 1

# consistent colors by 1-based label id
sorted_clusters = list(range(1, best_k + 1))
color_map = {c: base_cmap((c - 1) % base_cmap.N) for c in sorted_clusters}
legend_handles = [Patch(facecolor=color_map[c], edgecolor='k', label=f'Cluster {c}', alpha=0.7)
                  for c in sorted_clusters]

# features + cluster table for quick summaries
df = pd.DataFrame(features_flat, columns=feature_names)
df['cluster'] = labels_best
df['silhouette'] = sil_best
print('per-cluster means:\n', df.groupby('cluster')[feature_names + ['silhouette']].mean(), '\n')

# attach to metadata (master table for window-level analysis)
meta_df['cluster'] = labels_best

# save per-cluster means csv
per_cluster_means = (
    df.groupby('cluster')[feature_names + ['silhouette']]
      .mean()
      .reset_index()
      .sort_values('cluster')
)

per_cluster_means.to_csv(fig_dir / 'per_cluster_means.csv', index=False, float_format='%.3f')

# compute counts and proportions by station
counts = pd.crosstab(meta_df['station'], meta_df['cluster'])
props  = counts.div(counts.sum(axis=1), axis=0)
print('cluster counts:\n', counts, '\n')
print('cluster proportions:\n', props, '\n')

# stacked bar plot from largest to smallest
props = props.reindex(columns=sorted_clusters).copy()
props.index.name = None

ax = props.plot(kind='bar', stacked=True, figsize=(8, 4), 
                color=[color_map[c] for c in sorted_clusters],
                edgecolor='k', alpha=0.7
               )

# y-axis as percentages
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Proportion')
station = str(props.index[0])
ax.set_xlabel(None)
ax.tick_params(axis='x', labelbottom=False)
ax.set_title(f'Cluster Distribution')
plt.legend(handles=legend_handles, title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left')

fig = ax.get_figure()
fig.tight_layout()
fig.savefig(fig_dir / 'cluster_distribution.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)


# In[12]:


# ==================================
# Load and clean quake catalogue
# ==================================
# path
csv_path = Path('/home3/prwx27/1. MH-Dissertation/quake events database/quake-catalogue.csv')
if not csv_path.is_file():
    raise FileNotFoundError(f'CSV not found: {csv_path}')

# keep only needed columns
quake_df = (
    pd.read_csv(csv_path, dtype=str)
      .rename(columns=lambda c: c.lower().strip())
      .filter(['time','latitude','longitude','magnitude','eventlocationname'])
)

# restrict to events in japan
quake_df = quake_df[quake_df['eventlocationname'].str.contains('japan', case=False, na=False)]

# parse numerics and time, and round to 1 s to align with bins
quake_df[['latitude','longitude','magnitude']] = quake_df[['latitude','longitude','magnitude']].apply(
    pd.to_numeric, errors='coerce'
)
quake_df['time'] = pd.to_datetime(quake_df['time'], errors='coerce', utc=True).dt.round('s')

# drop any incomplete rows, then exact duplicates
quake_df = (
    quake_df.dropna(subset=['time','latitude','longitude','magnitude'])
            .drop_duplicates(subset=['time','latitude','longitude','magnitude'])
            .reset_index(drop=True)
)

# define week-1 window for downstream analysis
week1_start_label = '2016-04-18'
week1_end_label = '2016-04-24'

i0_week1, _ = sample_periods[week1_start_label]
_, i1_week1 = sample_periods[week1_end_label]

week1_start_ts = wave_file_start + pd.to_timedelta(i0_week1 / fs, unit='s')   
week1_end_ts = wave_file_start + pd.to_timedelta(i1_week1 / fs, unit='s')
qt0, qt1 = week1_start_ts, week1_end_ts

# check how many quakes fall inside week 1 before distance filtering
initial_mask = (quake_df['time'] >= qt0) & (quake_df['time'] <= qt1)
initial_quakes = quake_df.loc[initial_mask, 'time'].nunique()
print(f'Quakes in Week 1 before filtering: {initial_quakes}')


# In[13]:


# ===========================================
# Filtering quakes by distance to stations 
# ===========================================
# function to compute distance in km between station and quake location
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

# distance (km) from each event to each station
quake_df_raw = quake_df.copy()
dist_df = pd.DataFrame({
    st: haversine(
        float(station_coords[st][0]),
        float(station_coords[st][1]),
        quake_df_raw['latitude'].to_numpy(),
        quake_df_raw['longitude'].to_numpy()
    )
    for st in station_coords
}, index=quake_df_raw.index)

# pass rule per station: i) any magnitude within 100 km, ii) magnitude ≥ 2 within 200 km
m = quake_df_raw['magnitude'].to_numpy()
passes_by_station = {
    st: (((m >= 0) & (dist_df[st].to_numpy() <= 100)) |
         ((m >= 2) & (dist_df[st].to_numpy() <= 200)))
    for st in station_coords
}

# keep event if it passes at least one station
keep_any = np.zeros(len(quake_df_raw), dtype=bool)
for st in station_coords:
    keep_any |= passes_by_station[st]

quake_df_net = quake_df_raw.loc[keep_any].copy()
quake_df_net['distance_km'] = dist_df.loc[keep_any].min(axis=1).to_numpy()
quake_df_net['nearest_station'] = dist_df.loc[keep_any].idxmin(axis=1).to_numpy()
quake_df = quake_df_net.reset_index(drop=True)

# quakes split by station used for downstream analysis
quakes_by_station = {}
for st in station_coords:
    mask_st = passes_by_station[st]
    qst = quake_df_raw.loc[mask_st].copy()
    qst['distance_km'] = dist_df[st].loc[mask_st].to_numpy()
    quakes_by_station[st] = qst.reset_index(drop=True)


# In[14]:


# ==========================================================
# Cluster density + raw waveform + quake markers plot
# ==========================================================
# preprocess for plotting only: prepare raw waveform
with h5py.File(h5_path, 'r') as f:
    x_raw_week = f[station_a][0, i0_week1:i1_week1]
x_dt_week = detrend(x_raw_week, type='linear')
x_filt_week = filtfilt(b, a, x_dt_week)
x_norm_week = (x_filt_week - x_filt_week.mean()) / x_filt_week.std()
t_wave_days = np.arange(len(x_norm_week)) / (fs * 3600 * 24)

# prepare quake markers in this week
eq_a = quakes_by_station[station_a]
mask_week = (eq_a['time'] >= week1_start_ts) & (eq_a['time'] <= week1_end_ts)
eq_mags_wave = eq_a.loc[mask_week, 'magnitude'].to_numpy()
eq_pos_wave  = ((eq_a.loc[mask_week, 'time'] - week1_start_ts) / pd.Timedelta(days=1)).to_numpy()

# compute 10 min binned cluster counts ensuring an exact 10 min grid across the week
meta_df['time_bin'] = meta_df['timestamp'].dt.floor('10min')
full_bins = pd.date_range(start=week1_start_ts, end=week1_end_ts, freq='10min', tz='UTC', inclusive='left')
fixed_counts = (meta_df.groupby(['time_bin', 'cluster']).size().unstack(fill_value=0).reindex(index=full_bins, fill_value=0))

# order clusters by total size
size_order_desc = fixed_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
row_order = size_order_desc[::-1]   # rows: smallest cluster at top, largest at bottom
stack_order_top = size_order_desc[::-1]   # stacked bar plot: smallest cluster at bottom, largest at top

# convert bin times to day-fractions for compact x-axis
t_bin_days = (fixed_counts.index - week1_start_ts) / pd.Timedelta(days=1)

# quake markers aligned to the bin range
mask_bins = (eq_a['time'] >= week1_start_ts) & (eq_a['time'] <= week1_end_ts)
eq_times_bins = eq_a.loc[mask_bins, 'time']
eq_mags_bins = eq_a.loc[mask_bins, 'magnitude'].to_numpy()
eq_pos_bins = ((eq_times_bins - week1_start_ts) / pd.Timedelta(days=1)).to_numpy()

# plot: waveform row + one row per cluster + stacked density
n_clusters = len(row_order)
fig, axes = plt.subplots(n_clusters+2, 1, figsize=(14, 3*(n_clusters+2)), sharex=True)

# waveform row
ax0 = axes[0]
ax0.plot(t_wave_days, x_norm_week, color='k', linewidth=0.5)
ax0.set_ylabel('Normalised Amplitude')
ax0.grid(True)
ax0.set_xlim(0, 7)
if len(eq_pos_wave):
    ax0.scatter(eq_pos_wave, np.full_like(eq_pos_wave, -0.03, dtype=float), marker='v', s=24, color='red',
                transform=ax0.get_xaxis_transform(), clip_on=False, zorder=5)
    for xq, mq in zip(eq_pos_wave, eq_mags_wave):
        ax0.text(xq, -0.09, f'M{mq:.1f}', rotation=90, va='top', ha='center',
                 transform=ax0.get_xaxis_transform(), fontsize=7, color='red', clip_on=False)
        
# per-cluster rows (smallest at top)
for i, c in enumerate(row_order):
    ax = axes[i+1]
    y = fixed_counts[c].values
    ax.step(t_bin_days, y, where='post', color=color_map[c])
    ax.fill_between(t_bin_days, y, step='post', alpha=0.5, color=color_map[c])
    ax.set_ylabel(f'Cluster {c}\nCount')
    ax.set_ylim(0, 120)
    ax.set_xlim(0, 7)
    ax.grid(True)
    if len(eq_pos_bins):
        ax.scatter(eq_pos_bins, np.full_like(eq_pos_bins, -0.03, dtype=float),
                   marker='v', s=24, color='red',
                   transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)

# stacked density (largest at top)
ax_stack = axes[-1]
bottom = np.zeros_like(t_bin_days, dtype=float)
for c in stack_order_top:
    y = fixed_counts[c].values
    ax_stack.fill_between(t_bin_days, bottom, bottom + y,
                          step='post', alpha=0.8, color=color_map[c], label=f'Cluster {c}')
    bottom += y

ax_stack.set_ylabel('Stacked\nCount')
ax_stack.set_ylim(0, 120)
ax_stack.set_xlim(0, 7)
ax_stack.grid(True)

day_ticks = np.arange(0, 8, 1.0)
day_labels = [(week1_start_ts + pd.Timedelta(days=int(d))).strftime('%b %d') for d in day_ticks]
ax_stack.set_xticks(day_ticks)
ax_stack.set_xticklabels(day_labels)
ax_stack.tick_params(axis='x', which='major', length=9, width=1.4, labelsize=10)
ax_stack.grid(True, which='major', alpha=0.6)

hour_marks = np.array([6/24, 12/24, 18/24])
days = np.arange(7)
hour_ticks = (hour_marks[None, :] + days[:, None]).ravel()
hour_labels = (['06:00', '12:00', '18:00'] * len(days))
ax_stack.set_xticks(hour_ticks, minor=True)
ax_stack.set_xticklabels(hour_labels, minor=True)
ax_stack.tick_params(axis='x', which='minor', length=4, width=0.8, labelsize=8)
ax_stack.grid(True, which='minor', alpha=0.25)
legend_order_top_to_bottom = size_order_desc
legend_handles_stack = [Patch(facecolor=color_map[c], edgecolor='k', label=f'Cluster {c}', alpha=0.8)
                        for c in legend_order_top_to_bottom]
ax_stack.legend(handles=legend_handles_stack, title='Clusters',
                loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, borderaxespad=0.0)
axes[-1].set_xlabel('Date (UTC)')

fig.tight_layout()
fig.subplots_adjust(right=0.82, bottom=0.1)
fig.savefig(fig_dir / 'cluster_waveform_density.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)


# In[15]:


# =============================
# Labelled waveform overview 
# =============================
# waveform +/- 30 min around largest earthquake and zoom +/- 60 s around largest quake
# extract quakes 
eq_a = quakes_by_station[station_a]
eq_a = eq_a[(eq_a['time'] >= qt0) & (eq_a['time'] <= qt1)].copy()

if eq_a.empty:
    print(f'[{station_a}] No per-station quake events between {qt0} and {qt1}')
else:
    # select the largest magnitude event within the window
    largest_idx = eq_a['magnitude'].idxmax()
    largest_event = eq_a.loc[largest_idx]
    event_time = largest_event['time']
    event_mag = float(largest_event['magnitude'])

    # +/- 30 min overview
    half_w = pd.Timedelta('30min')
    win_start_lrg, win_end_lrg = event_time - half_w, event_time + half_w

    # convert to sample indices and clamp to file bounds
    sec0 = (win_start_lrg - wave_file_start) / pd.Timedelta('1s')
    sec1 = (win_end_lrg   - wave_file_start) / pd.Timedelta('1s')
    i0 = int(round(sec0 * fs))
    i1 = int(round(sec1 * fs))
    with h5py.File(h5_path, 'r') as f:
        nmax = f[station_a].shape[1]
        i0 = max(0, min(i0, nmax - 1))
        i1 = max(i0 + 1, min(i1, nmax))
        x_raw = f[station_a][0, i0:i1]

    # preprocess and normalise by max abs for visualisation
    x_dt = detrend(x_raw, type='linear')
    x_filt = filtfilt(b, a, x_dt)
    den = np.max(np.abs(x_filt))
    x_norm = x_filt / den if den != 0 else x_filt
    
    time_axis = wave_file_start + pd.to_timedelta((i0 + np.arange(i1 - i0)) / fs, unit='s')
    
    # list any quakes inside the +/- 30 min window (for markers)
    in_window = eq_a[(eq_a['time'] >= win_start_lrg) & (eq_a['time'] <= win_end_lrg)].copy()
    print(f'{len(in_window)} earthquakes in ±30 min window around M{event_mag:.1f}:')
    print(in_window[['time','magnitude']])

    # overlay coloured 5 s windows by cluster that fall in this span
    mask_w = (meta_df['timestamp'] >= win_start_lrg) & (meta_df['timestamp'] < win_end_lrg)
    windows_l = meta_df.loc[mask_w, ['abs_sample_idx','cluster']]

    fig, ax = plt.subplots(figsize=(14, 3))
    for _, row in windows_l.iterrows():
        ws = int(row.abs_sample_idx)
        start = max(ws, i0)
        end = min(ws + wlen, i1)
        if end > start:
            ax.plot(time_axis[start - i0:end - i0],
                    x_norm[start - i0:end - i0],
                    color=color_map.get(row.cluster, base_cmap((row.cluster - 1) % base_cmap.N)),
                    linewidth=0.8)

    # draw quake markers
    for _, ev in in_window.iterrows():
        t, m = ev['time'], float(ev['magnitude'])
        ax.axvline(t, color='black', linestyle='--', alpha=0.7)
        ax.text(t, 1.02, f'M{m:.1f}', rotation=90, va='bottom', ha='center',
                transform=ax.get_xaxis_transform(), color='black', fontsize=8)

    ax.set_xlim(win_start_lrg, win_end_lrg)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Normalised Amplitude')
    ax.grid(True)
    ticks = pd.date_range(win_start_lrg, win_end_lrg, freq='5min')
    ax.set_xticks(ticks)
    ax.set_xticklabels([t.strftime('%H:%M') for t in ticks])
    ax.legend(handles=legend_handles_stack, title='Clusters',
              loc='upper left', bbox_to_anchor=(1.02, 1.0),
              frameon=True, borderaxespad=0.0)
    fig.subplots_adjust(right=0.82)
    plt.tight_layout()
    fig.savefig(fig_dir / 'waveform_largest_eq_30min.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # +/- 60 s zoom
    half = pd.Timedelta('60s')
    ws, we = event_time - half, event_time + half

    sec0 = (ws - wave_file_start) / pd.Timedelta('1s')
    sec1 = (we - wave_file_start) / pd.Timedelta('1s')
    i0 = int(round(sec0 * fs))
    i1 = int(round(sec1 * fs))
    with h5py.File(h5_path, 'r') as f:
        nmax = f[station_a].shape[1]
        i0 = max(0, min(i0, nmax - 1))
        i1 = max(i0 + 1, min(i1, nmax))
        x_raw = f[station_a][0, i0:i1]

    x_dt = detrend(x_raw, type='linear')
    x_filt = filtfilt(b, a, x_dt)
    den = np.max(np.abs(x_filt))
    x_norm = x_filt / den if den != 0 else x_filt

    time_axis = wave_file_start + pd.to_timedelta((i0 + np.arange(i1 - i0)) / fs, unit='s')

    mask_w = (meta_df['timestamp'] >= ws) & (meta_df['timestamp'] < we)
    windows_c = meta_df.loc[mask_w, ['abs_sample_idx','cluster']]

    fig, ax = plt.subplots(figsize=(10, 3))
    for _, row in windows_c.iterrows():
        wsamp = int(row.abs_sample_idx)
        start = max(wsamp, i0)
        end = min(wsamp + wlen, i1)
        if end > start:
            ax.plot(time_axis[start - i0:end - i0],
                    x_norm[start - i0:end - i0],
                    color=color_map.get(row.cluster, base_cmap((row.cluster - 1) % base_cmap.N)),
                    linewidth=1.0)

    ax.set_xlim(ws, we)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Normalised Amplitude')
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(fig_dir / 'zoom_largest_eq_60s.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# In[16]:


# =================================
# Earthquakes by cluster summary
# =================================
summary = []
events_for_summary = (
    quakes_by_station[station_a]
      .loc[lambda d: (d['time'] >= qt0) & (d['time'] <= qt1)]
      .sort_values('time')
      .reset_index(drop=True)
)

for _, ev in events_for_summary.iterrows():
    t0 = ev['time']
    mag0 = ev['magnitude'] 

    # examine +/- 60 s around quake time
    win_start = t0 - pd.Timedelta('60s')
    win_end = t0 + pd.Timedelta('60s')

    # load that span and preprocess
    sec0 = (win_start - wave_file_start) / pd.Timedelta('1s')
    sec1 = (win_end   - wave_file_start) / pd.Timedelta('1s')
    i0 = int(round(sec0 * fs))
    i1 = int(round(sec1 * fs))

    with h5py.File(h5_path, 'r') as f:
        x_raw = f[station_a][0, i0:i1]
    x_dt = detrend(x_raw, type='linear')
    x_filt = filtfilt(b, a, x_dt)

    # normalized for peak location
    den = np.max(np.abs(x_filt))
    x_norm = x_filt / den if den != 0 else x_filt

    # absolute peak in this +/- 60 s window
    idx_peak = int(np.argmax(np.abs(x_norm)))
    peak_amp = float(np.max(np.abs(x_filt)))  # physical amplitude
    i_peak = i0 + idx_peak

    # find the 5 s window that contains the peak sample
    mask_win = (meta_df['abs_sample_idx'] <= i_peak) & (meta_df['abs_sample_idx'] + wlen > i_peak)
    window_row    = meta_df.loc[mask_win].iloc[0]
    quake_cluster = int(window_row['cluster'])

    summary.append({
        'quake_time': t0,
        'magnitude': mag0,
        'peak_amp': peak_amp,
        'cluster': quake_cluster
    })

df_quake_summary = pd.DataFrame(summary)

counts = df_quake_summary['cluster'].value_counts().sort_index()
print('number of quakes attributed to each cluster (by peak location):')
print(counts.to_frame(name='num_quakes'))

print('\nfull quake summary:')
print(df_quake_summary)

# save table
df_quake_summary_out = df_quake_summary.copy()
df_quake_summary_out['peak_amp']  = df_quake_summary_out['peak_amp'].round(3)
df_quake_summary_out = df_quake_summary_out.sort_values('quake_time')
df_quake_summary_out.to_csv(fig_dir / 'quakes_full_summary.csv', index=False)


# In[17]:


# =========================================
# Labelled waveform overview at midnight
# =========================================
# waveform +/- 10 min around midnight at start of day 3 (2016-04-20 00:00 UTC)
# clean reference point in the mid-week with quiet conditions
day3_label = '2016-04-20'
if day3_label not in sample_periods:
    raise KeyError(f"'{day3_label}' not found in sample_periods")

i0_day3, _ = sample_periods[day3_label]
center_ts = wave_file_start + pd.to_timedelta(i0_day3 / fs, unit='s')

# +/- 10 minutes
half_w = pd.Timedelta('10min')
win_start = center_ts - half_w
win_end = center_ts + half_w

sec0 = (win_start - wave_file_start) / pd.Timedelta('1s')
sec1 = (win_end - wave_file_start) / pd.Timedelta('1s')
i0 = int(round(sec0 * fs))
i1 = int(round(sec1 * fs))

with h5py.File(h5_path, 'r') as f:
    x_raw = f[station_a][0, i0:i1]
x_dt = detrend(x_raw, type='linear')
x_filt = filtfilt(b, a, x_dt)
x_norm = x_filt / np.max(np.abs(x_filt)) if np.max(np.abs(x_filt)) != 0 else x_filt

time_axis = wave_file_start + pd.to_timedelta((i0 + np.arange(i1 - i0)) / fs, unit='s')

mask_windows = (meta_df['timestamp'] >= win_start) & (meta_df['timestamp'] < win_end)
windows_mid = meta_df.loc[mask_windows, ['abs_sample_idx','cluster']]

fig, ax = plt.subplots(figsize=(14, 3))
for _, row in windows_mid.iterrows():
    wsamp = int(row['abs_sample_idx'])
    start = max(wsamp, i0)
    end = min(wsamp + wlen, i1)
    if end > start:
        rel0 = start - i0
        rel1 = end - i0
        ax.plot(time_axis[rel0:rel1], x_norm[rel0:rel1],
                color=color_map.get(row['cluster'], base_cmap((row['cluster'] - 1) % base_cmap.N)),
                linewidth=0.8)

ax.set_xlim(win_start, win_end)
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Normalised Amplitude')
ax.grid(True)
ticks = pd.date_range(win_start, win_end, freq='5min')
ax.set_xticks(ticks)                                
ax.set_xticklabels([t.strftime('%H:%M') for t in ticks])
ax.legend(handles=legend_handles_stack, title='Clusters',
          loc='upper left', bbox_to_anchor=(1.02, 1.0),
          frameon=True, borderaxespad=0.0)
fig.subplots_adjust(right=0.82)
plt.tight_layout()
fig.savefig(fig_dir / 'waveform_midnight_day3_10min.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)


# In[18]:


# ===========================
# Load and clean wind data
# ===========================
# path
wind_path = Path('/home3/prwx27/1. MH-Dissertation/wind speeds database from JMA/latest-jma-wind-data.csv')
if not wind_path.is_file():
    raise FileNotFoundError(f'CSV not found: {wind_path}')

# we keep only first two columns timestamp + wind speed (timestamps are in jst)
df_wind = pd.read_csv(wind_path, encoding='shift_jis', skiprows=5, usecols=[0, 1])
df_wind.columns = ['timestamp', 'wind_speed_mps']

# parse datetime jst to utc and then index
df_wind['timestamp'] = (
    pd.to_datetime(df_wind['timestamp'], format='%Y/%m/%d %H:%M:%S')
      .dt.tz_localize('Asia/Tokyo')
      .dt.tz_convert('UTC')
)

df_wind['wind_speed_mps'] = pd.to_numeric(df_wind['wind_speed_mps'], errors='coerce')
df_wind = df_wind.set_index('timestamp')[['wind_speed_mps']].copy()

# slice the analysis window
wind_hourly = df_wind.loc['2016-04-18':'2016-04-24']

# plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wind_hourly.index, wind_hourly['wind_speed_mps'], linewidth=1, color='tab:grey')
ax.set_xlim(wind_hourly.index.min(), wind_hourly.index.max())
ax.set_xlabel('Date (UTC)')
ax.set_ylabel('Wind Speed (m/s)', color='tab:grey')
ax.tick_params(axis='y', labelcolor='tab:grey')
ax.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)


# In[19]:


# ==================================================
# Resample clusters to hourly and merge with wind
# ==================================================
# comparison using clusters 2–4
cols = [c for c in (2, 3, 4) if c in fixed_counts.columns]
clusters_hourly = fixed_counts[cols].resample('h').sum()

# combine clusters + wind
df_all_hourly = clusters_hourly.join(wind_hourly[['wind_speed_mps']], how='inner').dropna()
print('Hourly combined data (clusters 2–3 vs wind speed):\n')
print(df_all_hourly.head())


# In[20]:


# ==============================================
# Correlation & cross-correlation (+/- 6 hrs)
# ==============================================
max_lag_hours = 6
results = []

# analyse clusters 2–4
targets = [2, 3, 4]
cluster_colours = {c: color_map.get(c, base_cmap((c - 1) % base_cmap.N)) for c in targets}

for cluster_num in targets:
    # extract series
    x = df_all_hourly[cluster_num]
    y = df_all_hourly['wind_speed_mps']

    # pearson correlation and significance at zero lag
    r0, p0 = pearsonr(x, y)

    # cross-correlation (demean, normalize by std, length)
    x_demeaned = x - x.mean()
    y_demeaned = y - y.mean()
    corr = correlate(x_demeaned, y_demeaned, mode='full')
    corr /= (np.std(x_demeaned) * np.std(y_demeaned) * len(x_demeaned))
    lags = correlation_lags(len(x_demeaned), len(y_demeaned), mode='full')

    # zoom to +/- max lag hours
    mask = (lags >= -max_lag_hours) & (lags <= max_lag_hours)
    lags_zoom = lags[mask]
    corr_zoom = corr[mask]
    
    # time series view
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(df_all_hourly.index, x, color=cluster_colours[cluster_num], label=f'Cluster {cluster_num} Counts')
    ax1.set_ylabel(f'Cluster {cluster_num} Counts (per Hour)', color=cluster_colours[cluster_num])
    ax1.tick_params(axis='y', labelcolor=cluster_colours[cluster_num])
    ax2 = ax1.twinx()
    ax2.plot(df_all_hourly.index, y, color='tab:grey', label='Wind Speed (m/s)')
    ax2.set_ylabel('Wind Speed (m/s)', color='tab:grey')
    ax2.tick_params(axis='y', labelcolor='tab:grey')
    ax1.set_xlabel('Date (UTC)') 
    xmin, xmax = df_all_hourly.index.min(), df_all_hourly.index.max()
    ax1.set_xlim(xmin, xmax)
    ax1.margins(x=0)
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(fig_dir / f'cluster{cluster_num}_vs_wind_hourly.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # cross-correlation view with signed lags in hours
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lags_zoom, corr_zoom, marker='o', color=cluster_colours[cluster_num])
    plt.xlabel('Lag (Hours)')
    plt.ylabel('Cross-correlation Coefficient')
    plt.axvline(0, color='gray', linestyle='--', label='Zero Lag')
    plt.legend()
    plt.grid(True)
    fig2.tight_layout()
    fig2.savefig(fig_dir / f'cluster{cluster_num}_vs_wind_xcorr_pm{max_lag_hours}h.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

    # record peak correlation within the window and its lag
    max_corr_idx = int(np.argmax(corr_zoom))
    best_lag = int(lags_zoom[max_corr_idx])
    best_corr = float(corr_zoom[max_corr_idx])
    
    sig_label = 'significant at 10%' if p0 < 0.10 else 'not significant at 10%'

    results.append({
        'cluster': cluster_num,
        'pearson_r': r0,
        'pearson_p': p0,
        'signif_10pct': sig_label,
        'max_crosscorr': best_corr,
        'best_lag_hr': best_lag
    })


# In[21]:


# summary table 
df_summary = pd.DataFrame(results)
print('Summary of cluster vs wind correlations (±6 h window):\n')
print(df_summary)


# In[22]:


# ====================================
# Alignment at best lag (Cluster 2)
# ====================================
# choose best cluster to align
cluster_to_align = 2
best_lag_hours = int(df_summary.loc[df_summary['cluster'] == cluster_to_align, 'best_lag_hr'].values[0])
print(f'Using best lag = {best_lag_hours} h for Cluster {cluster_to_align}')

# build aligned frame
df_aligned = df_all_hourly[[cluster_to_align, 'wind_speed_mps']].copy()
df_aligned = df_aligned.rename(columns={cluster_to_align: f'cluster{cluster_to_align}_counts'})
df_aligned['wind_shifted'] = df_aligned['wind_speed_mps'].shift(best_lag_hours)
df_aligned = df_aligned.dropna(subset=[f'cluster{cluster_to_align}_counts', 'wind_shifted'])

# visualise aligned series
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_aligned.index, df_aligned[f'cluster{cluster_to_align}_counts'],
         color=color_map.get(cluster_to_align, base_cmap((cluster_to_align - 1) % base_cmap.N)),
         label=f'Cluster {cluster_to_align} Counts')
ax1.set_ylabel(f'Cluster {cluster_to_align} Counts', color=cluster_colours[cluster_to_align])
ax1.tick_params(axis='y', labelcolor=cluster_colours[cluster_to_align])
ax2 = ax1.twinx()
ax2.plot(df_aligned.index, df_aligned['wind_shifted'],
         color='tab:grey', label=f'Wind Speed (shifted {best_lag_hours} h)')
ax2.set_ylabel('Wind Speed (m/s)', color='tab:grey')
ax2.tick_params(axis='y', labelcolor='tab:grey')
ax1.set_xlabel('Date (UTC)')
ax1.set_xlim(week1_start_ts, week1_end_ts)
ax2.set_xlim(week1_start_ts, week1_end_ts)
ax1.margins(x=0); ax2.margins(x=0)
plt.grid(True)
plt.tight_layout()
fig.savefig(fig_dir / f'cluster{cluster_to_align}_vs_wind_shifted_{best_lag_hours}h.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)

# check correlation after applying the lag shift
x_aligned = df_aligned[f'cluster{cluster_to_align}_counts']
y_aligned = df_aligned['wind_shifted']
r_shifted, p_shifted = pearsonr(x_aligned, y_aligned)

print(f'\nCorrelation after applying lag shift ({best_lag_hours} h):\n')
print(f'Pearson r = {r_shifted:.3f}, p_value = {p_shifted:.3e}')
if p_shifted < 0.05:
    print('Significant at the 5% level')
elif p_shifted < 0.10:
    print('Significant at the 10% level')
else:
    print('Not significant at the 10% level')


# In[23]:


# ======================== STATION B ===========================

# ===============================================
# B. Data preprocessing and feature extraction 
# ===============================================
# colour palette
base_cmap_B = plt.get_cmap('tab10')
start_idx_B = base_cmap_B.N - 1  # last colour in tab10

meta_records_B, features_list_B = [], []

with h5py.File(h5_path, 'r') as f:
    for period, (i0, i1) in sample_periods.items():
        nwin = (i1 - i0) // wlen

        # load, detrend, high-pass
        x_raw = f[station_b][0, i0:i1]
        x_dt  = detrend(x_raw, type='linear')
        x_f   = filtfilt(b, a, x_dt)

        # 5 s windows
        for wi in range(nwin):
            s0 = i0 + wi * wlen
            x_seg = x_f[wi*wlen : wi*wlen + wlen]

            feats = extract_features(x_seg, fs)
            features_list_B.append(feats)

            ts = wave_file_start + pd.Timedelta(seconds=s0/fs)
            meta_records_B.append({
                'period': period,
                'station': station_b,
                'window_index': wi,
                'abs_sample_idx': s0,
                'timestamp': ts
            })

meta_df_B = pd.DataFrame(meta_records_B)
features_flat_B = np.vstack(features_list_B)

print(meta_df_B.head(), '\n')
print(meta_df_B['station'].value_counts())
print('features_flat_B shape:', features_flat_B.shape)


# In[24]:


# =============================
# B. Standardisation and pca
# =============================
features_scaled_B = StandardScaler().fit_transform(features_flat_B)

pca_full_B   = PCA(random_state=random_state, whiten=False)
Xp_full_B    = pca_full_B.fit_transform(features_scaled_B)
explained_B  = pca_full_B.explained_variance_ratio_
cumulative_B = np.cumsum(explained_B)
print('Cumulative explained variance (Station B):', cumulative_B)

# Scree plot
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(1, len(explained_B) + 1)
ax.plot(xs, explained_B, 'ok--')
ax.set_xlabel('Principal Component Number')
ax.set_ylabel('Explained Variance')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.grid(True)
fig.tight_layout()
fig.savefig(fig_dir / f'scree_plot_{station_b}.png', dpi=600, bbox_inches='tight')
plt.show(); plt.close(fig)

# keep number of pcs explaining >90% of variance for clustering
target_variance = 0.90
n_components_B = int(np.searchsorted(cumulative_B, target_variance) + 1)
pca_n_B = PCA(n_components=n_components_B, random_state=random_state, whiten=False)
Xp_B = pca_n_B.fit_transform(features_scaled_B)
print('Xp_B shape:', Xp_B.shape)


# In[25]:


# ===========================================
# B. Optimal k selection for k-means model
# ===========================================
cluster_range = list(range(2, 21))
wss_B = []
for k in cluster_range:
    km = KMeans(
        n_clusters=k, init='k-means++', n_init=n_init,
        tol=kmeans_tol, max_iter=max_kmeans_iter,
        algorithm=kmeans_algorithm, random_state=random_state,
    )
    km.fit(Xp_B)
    wss_B.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cluster_range, wss_B, 'ok--')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Within-Cluster Sum of Squares')
ax.set_xticks(cluster_range)
fig.tight_layout()
fig.savefig(fig_dir / f'elbow_plot_{station_b}.png', dpi=600, bbox_inches='tight')
plt.show(); plt.close(fig)


# In[26]:


# silhouette around the elbow
sil_cluster_range_B = list(range(4, 8))
all_km_B, all_labels_B, all_sil_B, sil_avgs_B = {}, {}, {}, []

for k in sil_cluster_range_B:
    km = KMeans(
        n_clusters=k, init='k-means++', n_init=n_init,
        tol=kmeans_tol, max_iter=max_kmeans_iter,
        algorithm=kmeans_algorithm, random_state=random_state,
    )
    labels = km.fit_predict(Xp_B)
    sil_vals = silhouette_samples(Xp_B, labels)
    sil_avg  = silhouette_score(Xp_B, labels)

    all_km_B[k], all_labels_B[k], all_sil_B[k] = km, labels, sil_vals
    sil_avgs_B.append(sil_avg)

    # sort clusters by size and remap
    cluster_sizes = [np.sum(labels == i) for i in range(k)]
    sorted_clusters = sorted(range(k), key=lambda i: cluster_sizes[i], reverse=True)
    new_label_map  = {orig: new for new, orig in enumerate(sorted_clusters)}
    remapped_labels  = np.array([new_label_map[orig] for orig in labels])
    remapped_centers = np.array([km.cluster_centers_[orig] for orig in sorted_clusters])

    # silhouette + PC scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_title(f'Silhouette (k = {k}) — {station_b}')
    ax1.set_xlim([-0.2, 1]); ax1.set_xticks(np.arange(-0.2, 1.05, 0.1))
    ax1.set_ylim([0, len(Xp_B) + (k + 1) * 10])

    y_lower = 10
    for new_label, orig in enumerate(sorted_clusters):
        ith_vals = np.sort(sil_vals[labels == orig])
        y_upper = y_lower + len(ith_vals)
        color = base_cmap_B((start_idx_B - new_label) % base_cmap_B.N)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        x_text = -0.05 if new_label % 2 == 0 else 0.05
        ax1.text(x_text, y_lower + 0.5 * len(ith_vals), str(new_label + 1))
        y_lower = y_upper + 10
    ax1.axvline(sil_avg, color='red', linestyle='--')
    ax1.set_xlabel('Silhouette Score'); ax1.set_ylabel('Number of Samples')

    scatter_colors = [base_cmap_B((start_idx_B - l) % base_cmap_B.N) for l in remapped_labels]
    ax2.scatter(Xp_B[:, 0], Xp_B[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=scatter_colors)
    ax2.scatter(remapped_centers[:, 0], remapped_centers[:, 1],
                marker='o', c='white', s=200, edgecolor='k')
    for idx, c in enumerate(remapped_centers):
        ax2.scatter(c[0], c[1], marker=f'${idx+1}$', edgecolor='k')
    ax2.set_title(f'Cluster Visualisation (PC1 vs PC2) — {station_b}')

    fig.tight_layout()
    fig.savefig(fig_dir / f'silhouette_{station_b}_k{k}.png', dpi=600, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f'{station_b}: k = {k} → avg silhouette = {sil_avg:.4f}')

# summary plot of average silhouette across chosen k's
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sil_cluster_range_B, sil_avgs_B, 'ok--')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Average Silhouette Score')
ax.set_xticks(sil_cluster_range_B)
ax.grid(True)
fig.tight_layout()
fig.savefig(fig_dir / f'silhouette_vs_nclusters_{station_b}.png', dpi=600, bbox_inches='tight')
plt.show(); plt.close(fig)


# In[31]:


# =================================================
# Finalise k, relabel by size, attach to windows 
# =================================================
best_idx_B = int(np.argmax(sil_avgs_B))
best_k_B = sil_cluster_range_B[best_idx_B]

labels_orig_B = all_labels_B[best_k_B]
sil_best_B    = all_sil_B[best_k_B]
km_best_B     = all_km_B[best_k_B]

# remap labels so 1 = largest cluster 
cluster_sizes_B = pd.Series(labels_orig_B).value_counts().to_dict()
sorted_clusters_orig_B = sorted(cluster_sizes_B, key=lambda x: cluster_sizes_B[x], reverse=True)
new_label_map_B = {orig: new for new, orig in enumerate(sorted_clusters_orig_B)}
labels_best_B   = np.array([new_label_map_B[l] for l in labels_orig_B]) + 1 

# consistent colors by 1-based cluster id
sorted_clusters_B = list(range(1, best_k_B + 1))
color_map_B = {c: base_cmap_B((start_idx_B - (c - 1)) % base_cmap_B.N) for c in sorted_clusters_B}
legend_handles_B = [Patch(facecolor=color_map_B[c], edgecolor='k', label=f'Cluster {c}', alpha=0.7)
                    for c in sorted_clusters_B]

# per-cluster summary table
df_B = pd.DataFrame(features_flat_B, columns=feature_names)
df_B['cluster']    = labels_best_B
df_B['silhouette'] = sil_best_B
print(f'Per-cluster means — {station_b}:\n',
      df_B.groupby('cluster')[feature_names + ['silhouette']].mean(), '\n')

# attach to metadata
meta_df_B['cluster'] = labels_best_B

# save per-cluster means CSV
per_cluster_means_B = (
    df_B.groupby('cluster')[feature_names + ['silhouette']]
       .mean().reset_index().sort_values('cluster')
)
per_cluster_means_B.to_csv(fig_dir / f'per_cluster_means_{station_b}.csv',
                           index=False, float_format='%.3f')

# counts / proportions and stacked bar 
counts_B = pd.crosstab(meta_df_B['station'], meta_df_B['cluster'])
props_B  = counts_B.div(counts_B.sum(axis=1), axis=0)
print(f'Cluster counts by station — {station_b}:\n', counts_B, '\n')
print(f'Cluster proportions by station — {station_b}:\n', props_B, '\n')

props_B = props_B.reindex(columns=sorted_clusters_B).copy()
props_B.index.name = None

ax = props_B.plot(kind='bar', stacked=True, figsize=(8, 4),
                  color=[color_map_B[c] for c in sorted_clusters_B],
                  edgecolor='k', alpha=0.7)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Proportion')
ax.set_xlabel(None)
ax.tick_params(axis='x', labelbottom=False)
ax.set_title(f'Cluster Distribution — {station_b}')
plt.legend(handles=legend_handles_B, title='Clusters',
           bbox_to_anchor=(1.02, 1), loc='upper left')
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(fig_dir / f'cluster_distribution_{station_b}.png', dpi=600, bbox_inches='tight')
plt.show(); plt.close(fig)


# In[28]:


# ======================================================
# Cluster density + raw waveform + quake markers plot
# ======================================================
with h5py.File(h5_path, 'r') as f:
    x_raw_week_B = f[station_b][0, i0_week1:i1_week1]

x_dt_week_B   = detrend(x_raw_week_B, type='linear')
x_filt_week_B = filtfilt(b, a, x_dt_week_B)
x_norm_week_B = (x_filt_week_B - x_filt_week_B.mean()) / x_filt_week_B.std()
t_wave_days_B = np.arange(len(x_norm_week_B)) / (fs * 3600 * 24)

# quake markers for this station in Week 1
eqB = quakes_by_station[station_b]
mask_week_B    = (eqB['time'] >= week1_start_ts) & (eqB['time'] <= week1_end_ts)
eq_mags_wave_B = eqB.loc[mask_week_B, 'magnitude'].to_numpy()
eq_pos_wave_B  = ((eqB.loc[mask_week_B, 'time'] - week1_start_ts) / pd.Timedelta(days=1)).to_numpy()

# 10-min fixed grid, counts by cluster
meta_df_B['time_bin'] = meta_df_B['timestamp'].dt.floor('10min')
full_bins_B = pd.date_range(start=week1_start_ts, end=week1_end_ts,
                            freq='10min', tz='UTC', inclusive='left')
fixed_counts_B = (
    meta_df_B.groupby(['time_bin', 'cluster']).size().unstack(fill_value=0)
             .reindex(index=full_bins_B, fill_value=0)
)

# order rows
size_order_desc_B = fixed_counts_B.sum(axis=0).sort_values(ascending=False).index.tolist()
row_order_B       = size_order_desc_B[::-1]
stack_order_top_B = size_order_desc_B[::-1]

t_bin_days_B = (fixed_counts_B.index - week1_start_ts) / pd.Timedelta(days=1)

# quake dots aligned to bin axis
eq_times_bins_B = eqB.loc[mask_week_B, 'time']
eq_pos_bins_B   = ((eq_times_bins_B - week1_start_ts) / pd.Timedelta(days=1)).to_numpy()

n_clusters_B = len(row_order_B)
fig, axes = plt.subplots(n_clusters_B + 2, 1, figsize=(14, 3*(n_clusters_B+2)), sharex=True)

# waveform row
ax0 = axes[0]
ax0.plot(t_wave_days_B, x_norm_week_B, color='k', linewidth=0.5)
ax0.set_ylabel('Normalised Amplitude')
ax0.grid(True); ax0.set_xlim(0, 7)
if len(eq_pos_wave_B):
    ax0.scatter(eq_pos_wave_B, np.full_like(eq_pos_wave_B, -0.03, dtype=float),
                marker='v', s=24, color='red',
                transform=ax0.get_xaxis_transform(), clip_on=False, zorder=5)
    for xq, mq in zip(eq_pos_wave_B, eq_mags_wave_B):
        ax0.text(xq, -0.09, f'M{float(mq):.1f}', rotation=90, va='top', ha='center',
                 transform=ax0.get_xaxis_transform(), fontsize=7, color='red', clip_on=False)

# per-cluster rows (smallest at top)
global_max_B = fixed_counts_B.to_numpy().max()
for i, c in enumerate(row_order_B):
    ax = axes[i+1]
    y = fixed_counts_B[c].values
    ax.step(t_bin_days_B, y, where='post', color=color_map_B[c])
    ax.fill_between(t_bin_days_B, y, step='post', alpha=0.5, color=color_map_B[c])
    ax.set_ylabel(f'Cluster {c}\nCount')
    ax.set_ylim(0, max(120, global_max_B))
    ax.set_xlim(0, 7)
    ax.grid(True)
    if len(eq_pos_bins_B):
        ax.scatter(eq_pos_bins_B, np.full_like(eq_pos_bins_B, -0.03, dtype=float),
                   marker='v', s=24, color='red',
                   transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)

# stacked density (largest at top)
ax_stack = axes[-1]
bottom = np.zeros_like(t_bin_days_B, dtype=float)
for c in stack_order_top_B:
    y = fixed_counts_B[c].values
    ax_stack.fill_between(t_bin_days_B, bottom, bottom + y,
                          step='post', alpha=0.8, color=color_map_B[c], label=f'Cluster {c}')
    bottom += y

ax_stack.set_ylabel('Stacked\nCount')
ax_stack.set_ylim(0, max(120, global_max_B))
ax_stack.set_xlim(0, 7)
ax_stack.grid(True)

# day ticks
day_ticks = np.arange(0, 8, 1.0)
day_labels = [(week1_start_ts + pd.Timedelta(days=int(d))).strftime('%b %d') for d in day_ticks]
ax_stack.set_xticks(day_ticks); ax_stack.set_xticklabels(day_labels)
ax_stack.tick_params(axis='x', which='major', length=9, width=1.4, labelsize=10)
ax_stack.grid(True, which='major', alpha=0.6)

hour_marks = np.array([6/24, 12/24, 18/24])
days = np.arange(7)
hour_ticks = (hour_marks[None, :] + days[:, None]).ravel()
hour_labels = (['06:00', '12:00', '18:00'] * len(days))
ax_stack.set_xticks(hour_ticks, minor=True)
ax_stack.set_xticklabels(hour_labels, minor=True)
ax_stack.tick_params(axis='x', which='minor', length=4, width=0.8, labelsize=8)
ax_stack.grid(True, which='minor', alpha=0.25)

legend_order_top_to_bottom_B = size_order_desc_B
legend_handles_stack_B = [Patch(facecolor=color_map_B[c], edgecolor='k', label=f'Cluster {c}', alpha=0.8)
                          for c in legend_order_top_to_bottom_B]
ax_stack.legend(handles=legend_handles_stack_B, title=f'Clusters',
                loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, borderaxespad=0.0)

axes[-1].set_xlabel('Date (UTC)')
fig.tight_layout()
fig.subplots_adjust(right=0.82, bottom=0.1)
fig.savefig(fig_dir / f'cluster_waveform_density_{station_b}.png', dpi=600, bbox_inches='tight')
plt.show(); plt.close(fig)


# In[29]:


# ============================================
# Cumulative counts per cluster per station
# ============================================
base_cmap = plt.get_cmap('tab10')

clusters_A = sorted(pd.unique(meta_df['cluster']))
clusters_B = sorted(pd.unique(meta_df_B['cluster']))

# plot
fig, ax = plt.subplots(figsize=(11, 6))
y_max = 0

# Station A (IMRH) lines 
for c in clusters_A:
    times = (
        meta_df.loc[
            (meta_df['cluster'] == c) &
            (meta_df['timestamp'] >= week1_start_ts) &
            (meta_df['timestamp'] <  week1_end_ts),
            'timestamp'
        ].sort_values()
    )
    n = len(times)
    if n == 0:
        continue
    y = np.arange(1, n + 1, dtype=float)
    t = pd.DatetimeIndex(times)
    t_plot = pd.DatetimeIndex([week1_start_ts]).append(t).append(pd.DatetimeIndex([week1_end_ts]))
    y_plot = np.r_[0.0, y, float(n)]        
    y_max = max(y_max, n)
    col = color_map.get(c, base_cmap((c - 1) % base_cmap.N))
    ax.step(t_plot, y_plot, where='post', color=col, linewidth=2.0, linestyle='-')

# Station B (UWEH) lines
for c in clusters_B:
    timesB = (
        meta_df_B.loc[
            (meta_df_B['cluster'] == c) &
            (meta_df_B['timestamp'] >= week1_start_ts) &
            (meta_df_B['timestamp'] <  week1_end_ts),
            'timestamp'
        ].sort_values()
    )
    nB = len(timesB)
    if nB == 0:
        continue
    yB = np.arange(1, nB + 1, dtype=float)       
    tB = pd.DatetimeIndex(timesB)
    tB_plot = pd.DatetimeIndex([week1_start_ts]).append(tB).append(pd.DatetimeIndex([week1_end_ts]))
    yB_plot = np.r_[0.0, yB, float(nB)]
    y_max = max(y_max, nB)
    colB = color_map_B.get(c, base_cmap((c - 1) % base_cmap.N))
    ax.step(tB_plot, yB_plot, where='post', color=colB, linewidth=2.0, linestyle='--')

# axes + ticks
ax.set_xlim(week1_start_ts, week1_end_ts)
ax.set_ylim(0, y_max if y_max > 0 else 1)
ax.margins(y=0)  
ax.set_ylabel('Cumulative Count')
ax.set_xlabel('Date (UTC)')
ax.grid(True, alpha=0.4)

day_ticks = pd.date_range(week1_start_ts, week1_end_ts, freq='1D', tz='UTC')
ax.set_xticks(day_ticks)
ax.set_xticklabels([d.strftime('%b %d') for d in day_ticks])

# legends: cluster colours per station
handles_A = [Line2D([0], [0], color=color_map.get(c, base_cmap((c - 1) % base_cmap.N)),
                    lw=2, linestyle='-', label=f'Cluster {c}A') for c in clusters_A]
handles_B = [Line2D([0], [0], color=color_map_B.get(c, base_cmap((c - 1) % base_cmap.N)),
                    lw=2, linestyle='--', label=f'Cluster {c}B') for c in clusters_B]

legA = ax.legend(handles=handles_A, title=f'Station SAGH02', loc='upper left',
                 bbox_to_anchor=(1.02, 1.0))
ax.add_artist(legA)
ax.legend(handles=handles_B, title=f'Station KMMH13', loc='upper left',
          bbox_to_anchor=(1.02, 0.65))

fig.tight_layout()
fig.subplots_adjust(right=0.82)
fig.savefig(fig_dir / 'cumulative_detections_IMRH_vs_UWEH_counts.png', dpi=600, bbox_inches='tight')
plt.show()
plt.close(fig)

