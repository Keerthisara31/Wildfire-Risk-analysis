import streamlit as st
import pandas as pd
import torch
from torch_geometric.data import Data
from gcn_model import GCN
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fire Risk Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("../models/scaler_fire.pkl")
        model = GCN(in_channels=4, hidden_channels=16, out_channels=2)
        model.load_state_dict(torch.load("../models/gcn_fire_model.pt", map_location=torch.device('cpu')))
        model.eval()
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

@st.cache_data
def preprocess_data(df):
    df = df[df['type'].isin([0, 2])].copy()
    df['type'] = df['type'].map({0: 0, 2: 1})
    df = df.dropna(subset=['brightness', 'frp', 'bright_t31', 'daynight', 'latitude', 'longitude'])
    df['daynight'] = df['daynight'].map({'D': 1, 'N': 0})
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
    return df

def create_graph_data(df, scaler, distance_threshold=0.5):
    features = df[['brightness', 'frp', 'bright_t31', 'daynight']].values
    features = scaler.transform(features)
    coords = df[['latitude', 'longitude']].values

    dists = distance_matrix(coords, coords)
    edge_index = []
    for i in range(len(dists)):
        for j in range(len(dists)):
            if i != j and dists[i, j] < distance_threshold:
                edge_index.append([i, j])

    if not edge_index:
        for i in range(min(len(coords)-1, 10)):
            edge_index.append([i, i+1])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index), coords

def create_enhanced_map(coords, predictions, df):
    center_lat, center_lon = coords[:, 0].mean(), coords[:, 1].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)

    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png',
        attr='© OpenStreetMap contributors, © CartoDB',
        name='Light', control=True).add_to(m)
    folium.TileLayer(
        tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        name='Terrain', control=True).add_to(m)

    fire_group = folium.FeatureGroup(name='🔥 Fire Risk Points')
    safe_group = folium.FeatureGroup(name='✅ Safe Points')

    for i in range(len(coords)):
        lat, lon = coords[i][0], coords[i][1]
        is_fire = predictions[i] == 1
        brightness = df.iloc[i]['brightness']
        frp = df.iloc[i]['frp']
        daynight = 'Day' if df.iloc[i]['daynight'] == 1 else 'Night'

        popup_text = (
            f"<b>Prediction:</b> {'🔥 Fire Risk' if is_fire else '✅ Safe'}<br>"
            f"<b>Brightness:</b> {brightness}<br>"
            f"<b>FRP:</b> {frp}<br>"
            f"<b>Time:</b> {daynight}<br>"
            f"<b>Coordinates:</b> {lat:.4f}, {lon:.4f}"
        )

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=6 if is_fire else 4,
            color='darkred' if is_fire else 'darkblue',
            fill=True,
            fill_color='red' if is_fire else 'blue',
            fill_opacity=0.8 if is_fire else 0.6,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip="🔥 Fire Risk" if is_fire else "✅ Safe"
        )
        (fire_group if is_fire else safe_group).add_child(marker)

    m.add_child(fire_group)
    m.add_child(safe_group)

    fire_coords = coords[predictions == 1]
    if len(fire_coords) > 0:
        heat_data = [[lat, lon, 1] for lat, lon in fire_coords]
        heatmap = folium.FeatureGroup(name='🌡️ Fire Risk Heatmap')
        HeatMap(heat_data, radius=15, blur=10, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}).add_to(heatmap)
        m.add_child(heatmap)

    folium.LayerControl().add_to(m)
    return m

def create_analytics_dashboard(df, predictions):
    pred_labels = ['Safe' if p == 0 else 'Fire Risk' for p in predictions]
    df['Prediction'] = pred_labels

    pie_fig = px.pie(df, names='Prediction', title='Fire Risk Distribution', color='Prediction',
                     color_discrete_map={'Safe': 'blue', 'Fire Risk': 'red'})

    scatter_fig = px.scatter(df, x='brightness', y='frp', color='Prediction',
                             color_discrete_map={'Safe': 'blue', 'Fire Risk': 'red'},
                             title='Brightness vs Fire Radiative Power')

    bar_fig = px.histogram(df, x='Prediction', y='bright_t31', color='Prediction',
                           color_discrete_map={'Safe': 'blue', 'Fire Risk': 'red'},
                           title='Brightness T31 Distribution by Prediction', barmode='group')

    return pie_fig, scatter_fig, bar_fig

def main():
    st.title("🔥 Advanced Fire Risk GCN Prediction System")
    st.markdown("Upload FIRMS satellite data to predict fire risk using Graph Convolutional Networks")

    model, scaler = load_models()
    uploaded_file = st.file_uploader("Upload FIRMS CSV", type=['csv'])

    show_analytics = st.toggle("📊 Show Analytics Dashboard", value=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df_processed = preprocess_data(df)
        if df_processed.empty:
            st.warning("No valid rows in the uploaded file.")
            return

        data, coords = create_graph_data(df_processed, scaler)
        with torch.no_grad():
            _, preds = model(data).max(dim=1)
        preds = preds.numpy()

        fire_count = int((preds == 1).sum())
        safe_count = int((preds == 0).sum())
        total = len(preds)
        fire_pct = (fire_count / total) * 100 if total > 0 else 0

        st.metric("🔥 Fire Risk Points", fire_count)
        st.metric("✅ Safe Points", safe_count)
        st.metric("📊 Total Points", total)
        st.metric("⚠️ Risk %", f"{fire_pct:.1f}%")

        st.subheader("🗺️ Interactive Map")
        fmap = create_enhanced_map(coords, preds, df_processed)
        st_folium(fmap, width=1200, height=600)

        if show_analytics:
            st.subheader("📈 Analytics Dashboard")
            fig_pie, fig_scatter, fig_bar = create_analytics_dashboard(df_processed.copy(), preds)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.subheader("📊 Summary Statistics")
                summary_stats = df_processed.groupby(preds).agg({
                    'brightness': ['mean', 'std'],
                    'frp': ['mean', 'std'],
                    'bright_t31': ['mean', 'std']
                }).round(2)
                st.dataframe(summary_stats)

        st.subheader("💾 Download Results")
        results_df = df_processed.copy()
        results_df['fire_risk_prediction'] = preds
        results_df['risk_label'] = results_df['fire_risk_prediction'].map({1: 'Fire Risk', 0: 'Safe'})

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download predictions as CSV",
            data=csv,
            file_name=f"fire_risk_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()