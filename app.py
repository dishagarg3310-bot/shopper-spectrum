import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load saved models
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
item_similarity_df = joblib.load('item_similarity.pkl')
product_list = joblib.load('product_list.pkl')

# Segment mapping
rfm_segments = pd.read_csv('rfm_segments.csv')
segment_map = rfm_segments.groupby('Cluster')['Segment'].first().to_dict()

# App Config
st.set_page_config(page_title='Shopper Spectrum', page_icon='🛒', layout='wide')

# Custom CSS
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 2px solid #e94560;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Title */
    h1 { 
        color: #f5a623 !important;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 8px rgba(245,166,35,0.4);
    }
    h2, h3 { 
        color: #e94560 !important; 
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #f5a623);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233,69,96,0.4);
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(233,69,96,0.6);
    }

    /* Input boxes */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid #e94560 !important;
        border-radius: 10px !important;
    }

    /* Success box */
    .stSuccess {
        background: rgba(46,204,113,0.2) !important;
        border-left: 4px solid #2ecc71 !important;
        border-radius: 10px !important;
    }

    /* Info box */
    .stInfo {
        background: rgba(52,152,219,0.2) !important;
        border-left: 4px solid #3498db !important;
        border-radius: 10px !important;
    }

    /* Cards for recommendations */
    .rec-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(233,69,96,0.4);
        border-radius: 12px;
        padding: 12px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        background: rgba(233,69,96,0.15);
        border-color: #e94560;
        transform: translateX(5px);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(245,166,35,0.3);
    }

    /* Table */
    table {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
    }
    th {
        background: rgba(233,69,96,0.3) !important;
        color: white !important;
    }
    td { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🛒 Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio('Navigate', 
                         ['🏠 Home', 
                          '📦 Product Recommendation', 
                          '👥 Customer Segmentation'])
st.sidebar.markdown("---")
st.sidebar.markdown("*E-Commerce Analytics Tool*")

# ── HOME ──
if page == '🏠 Home':
    st.title('🛒 Shopper Spectrum')
    st.subheader('Customer Segmentation & Product Recommendation System')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2>📦 Product Recommender</h2>
            <p style='color:#ccc;'>Enter any product name and instantly get 
            Top 5 similar products using Collaborative Filtering</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h2>👥 Customer Segmentation</h2>
            <p style='color:#ccc;'>Enter RFM values and predict which 
            customer segment they belong to using KMeans Clustering</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Customer Segments")
    st.markdown("""
    | Segment | Description | Strategy |
    |---|---|---|
    | 🟢 High-Value | Recent, frequent, high spenders | VIP Loyalty Programs |
    | 🔵 Regular | Steady purchasers, moderate spend | Upsell & Referral |
    | 🟠 Occasional | Rare buyers, low spend | Re-engagement Campaigns |
    | 🔴 At-Risk | Haven't purchased in a long time | Win-back Discounts |
    """)

# ── PRODUCT RECOMMENDATION ──
elif page == '📦 Product Recommendation':
    st.title('📦 Product Recommender')
    st.markdown("*Powered by Item-Based Collaborative Filtering & Cosine Similarity*")
    st.markdown("---")

    product_input = st.text_input('🔍 Enter Product Name',
                                   placeholder='e.g. WHITE HANGING HEART T-LIGHT HOLDER')

    if st.button('✨ Get Recommendations'):
        if product_input.strip() == '':
            st.warning('Please enter a product name!')
        else:
            product_name = product_input.strip().upper()

            if product_name not in item_similarity_df.index:
                matches = [p for p in item_similarity_df.index if product_name in p]
                if not matches:
                    st.error(f'❌ Product "{product_name}" not found!')
                    st.stop()
                product_name = matches[0]
                st.info(f'Showing results for: **{product_name}**')

            sim_scores = item_similarity_df[product_name].drop(product_name)
            top5 = sim_scores.sort_values(ascending=False).head(5)

            st.success('✅ Top 5 Recommended Products:')
            
            rank_colors = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
            for i, (product, score) in enumerate(top5.items()):
                st.markdown(f"""
                <div class='rec-card'>
                    {rank_colors[i]} <b>{product}</b> 
                    <span style='float:right; color:#f5a623;'>
                    Similarity: {score:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

            # Chart
            st.markdown("---")
            st.subheader('📊 Similarity Scores')
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')
            bars = ax.barh(list(top5.index), list(top5.values),
                           color=['#e94560','#f5a623','#2ecc71','#3498db','#9b59b6'])
            ax.set_xlabel('Cosine Similarity Score', color='white')
            ax.set_title(f'Top 5 Similar to: {product_name[:40]}',
                         fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#444')
            ax.spines['left'].set_color('#444')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
            for bar, val in zip(bars, top5.values):
                ax.text(bar.get_width() + 0.005, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', color='white', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

# ── CUSTOMER SEGMENTATION ──
elif page == '👥 Customer Segmentation':
    st.title('👥 Customer Segmentation')
    st.markdown("*Powered by RFM Analysis & KMeans Clustering*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input('📅 Recency (days since last purchase)',
                                   min_value=0, value=30)
    with col2:
        frequency = st.number_input('🔁 Frequency (number of purchases)',
                                     min_value=1, value=5)
    with col3:
        monetary = st.number_input('💰 Monetary (total spend £)',
                                    min_value=0.0, value=500.0)

    if st.button('🎯 Predict Segment'):
        log_freq = np.log1p(frequency)
        log_mon  = np.log1p(monetary)
        input_scaled = scaler.transform([[recency, log_freq, log_mon]])
        cluster  = kmeans.predict(input_scaled)[0]
        segment  = segment_map.get(cluster, 'Unknown')

        seg_colors = {
            'High-Value': '#2ecc71',
            'Regular'   : '#3498db',
            'Occasional': '#e67e22',
            'At-Risk'   : '#e74c3c'
        }
        emojis = {
            'High-Value': '🟢',
            'Regular'   : '🔵',
            'Occasional': '🟠',
            'At-Risk'   : '🔴'
        }
        color   = seg_colors.get(segment, '#fff')
        emoji   = emojis.get(segment, '⚪')

        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.08); 
                    border-left: 5px solid {color};
                    border-radius: 12px; 
                    padding: 20px; 
                    margin: 10px 0;'>
            <h2 style='color:{color};'>{emoji} {segment} Customer</h2>
        </div>
        """, unsafe_allow_html=True)

        descriptions = {
            'High-Value': '⭐ VIP customer! Recent, frequent buyer with high spending. Offer exclusive loyalty rewards and early access to new products.',
            'Regular'   : '👍 Loyal customer! Steady purchaser. Target with upsell opportunities and referral programs.',
            'Occasional': '💡 Potential customer! Rare buyer. Send personalized re-engagement campaigns and seasonal offers.',
            'At-Risk'   : '⚠️ Needs attention! Inactive customer. Send win-back campaigns with heavy discounts immediately!'
        }
        st.info(descriptions.get(segment, ''))

        # RFM Chart
        st.markdown("---")
        st.subheader('📊 Customer Profile')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        fig.patch.set_facecolor('#1a1a2e')
        metrics     = ['Recency\n(days)', 'Frequency\n(orders)', 'Monetary\n(£)']
        values      = [recency, frequency, monetary]
        bar_colors  = ['#e94560', '#f5a623', '#2ecc71']

        for i, (metric, val, c) in enumerate(zip(metrics, values, bar_colors)):
            axes[i].set_facecolor('#16213e')
            axes[i].bar(['You'], [val], color=c, alpha=0.9, width=0.4)
            axes[i].set_title(metric, fontweight='bold', color='white')
            axes[i].tick_params(colors='white')
            for spine in axes[i].spines.values():
                spine.set_color('#444')
            axes[i].text(0, val * 0.5, str(round(val, 1)),
                         ha='center', va='center',
                         color='white', fontweight='bold', fontsize=13)

        plt.suptitle(f'Customer Profile — {segment} Segment',
                     fontweight='bold', color='white', fontsize=13)
        fig.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig)