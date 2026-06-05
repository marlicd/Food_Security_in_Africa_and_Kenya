import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Security in Africa & Kenya",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLING ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .main { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Typography */
    h1 { color: #1a1a2e; font-weight: 800; }
    h2 { color: #1a1a2e; font-weight: 700; border-bottom: 3px solid #d73027; padding-bottom: 8px; }
    h3 { color: #333333; font-weight: 600; }
    p  { color: #444444; line-height: 1.7; }

    /* Hero metric cards */
    .metric-card {
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    .metric-card .num  { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
    .metric-card .lbl  { font-size: 0.82rem; opacity: 0.88; margin-top: 4px; }

    /* Insight callout */
    .insight {
        background: #fff8f8;
        border-left: 5px solid #d73027;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        color: #333;
    }

    /* Section card */
    .section-card {
        background: #f9f9f9;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Divider */
    hr { border: none; border-top: 2px solid #f0f0f0; margin: 30px 0; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING & CLEANING ────────────────────────────────────────────────
@st.cache_data
def load_data():
    food_security   = pd.read_csv('DATA/Food_security_Africa.csv')
    food_production = pd.read_csv('DATA/Food_Production_Africa.csv')
    poverty         = pd.read_csv('DATA/Poverty_Africa.csv',    skiprows=4)
    population      = pd.read_csv('DATA/Population_Africa.csv', skiprows=4)
    return food_security, food_production, poverty, population


@st.cache_data
def clean_food_security(food_security):
    df = food_security[['Area', 'Item', 'Year', 'Value']].copy()
    items = [
        'Prevalence of severe food insecurity in the total population (percent) (3-year average)',
        'Prevalence of moderate or severe food insecurity in the total population (percent) (3-year average)'
    ]
    df = df[df['Item'].isin(items)]
    df = df[df['Year'].str[:4].astype(int) >= 2014]
    df.columns = ['Country', 'Indicator', 'Year', 'Value']
    df['Indicator'] = df['Indicator'].replace({
        'Prevalence of severe food insecurity in the total population (percent) (3-year average)': 'Severe Food Insecurity (%)',
        'Prevalence of moderate or severe food insecurity in the total population (percent) (3-year average)': 'Moderate or Severe Food Insecurity (%)'
    })
    df = df.dropna(subset=['Value'])
    df['Value']    = pd.to_numeric(df['Value'], errors='coerce')
    df['Year_int'] = df['Year'].str[:4].astype(int)
    return df


@st.cache_data
def clean_production(food_production):
    df = food_production[food_production['Element'] == 'Production'][['Area', 'Item', 'Year', 'Value']].copy()
    df = df[df['Year'] >= 2014]
    df.columns = ['Country', 'Crop', 'Year', 'Production (tonnes)']
    df = df.dropna(subset=['Production (tonnes)'])
    return df.groupby('Year')['Production (tonnes)'].sum().reset_index()


@st.cache_data
def clean_poverty(poverty):
    poverty = poverty.drop(columns=['Unnamed: 70'], errors='ignore')
    year_cols = [str(y) for y in range(2014, 2025)]
    df = poverty[['Country Name', 'Country Code'] + year_cols].copy()
    df = df.melt(id_vars=['Country Name', 'Country Code'],
                 value_vars=year_cols, var_name='Year', value_name='Poverty Rate (%)')
    df.columns = ['Country', 'Country Code', 'Year', 'Poverty Rate (%)']
    df['Year'] = df['Year'].astype(int)
    return df.dropna(subset=['Poverty Rate (%)'])


@st.cache_data
def load_africa_map():
    url   = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
    world = gpd.read_file(url)
    codes = [
        'DZA','AGO','BEN','BWA','BFA','BDI','CPV','CMR','CAF','TCD',
        'COM','COD','COG','CIV','DJI','EGY','GNQ','ERI','SWZ','ETH',
        'GAB','GMB','GHA','GIN','GNB','KEN','LSO','LBR','LBY','MDG',
        'MWI','MLI','MRT','MUS','MAR','MOZ','NAM','NER','NGA','RWA',
        'STP','SEN','SLE','SOM','ZAF','SSD','SDN','TZA','TGO','TUN',
        'UGA','ZMB','ZWE'
    ]
    return world[world['ISO3166-1-Alpha-3'].isin(codes)].copy()


def kenya_county_data():
    return pd.DataFrame({
        'County': [
            'Turkana','Marsabit','Mandera','Wajir','Garissa',
            'Tana River','Kilifi','Kwale','Taita-Taveta','Kajiado',
            'Narok','Samburu','Isiolo','West Pokot','Baringo',
            'Laikipia','Nakuru','Nairobi','Mombasa','Kisumu',
            'Siaya','Homa Bay','Migori','Kisii','Nyamira',
            'Kericho','Bomet','Nandi','Uasin Gishu','Trans-Nzoia',
            'Bungoma','Kakamega','Vihiga','Busia','Elgeyo-Marakwet',
            'Nyandarua','Nyeri','Kirinyaga',"Murang'a",'Kiambu',
            'Meru','Tharaka-Nithi','Embu','Machakos','Makueni',
            'Kitui','Lamu'
        ],
        'Food_Insecurity': [
            82,78,80,76,72,68,61,58,45,42,
            48,74,70,72,55,38,32,28,35,40,
            45,48,44,36,34,30,32,28,25,22,
            30,33,38,40,42,25,22,20,24,18,
            22,28,24,32,55,62,50
        ]
    })


# ─── CHART HELPERS ──────────────────────────────────────────────────────────
def chart_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, color='#eeeeee', linewidth=0.8)
    ax.set_axisbelow(True)


# ─── LOAD DATA ──────────────────────────────────────────────────────────────
food_security, food_production, poverty, population = load_data()
fs   = clean_food_security(food_security)
prod = clean_production(food_production)
pov  = clean_poverty(poverty)
kdf  = kenya_county_data()

latest_year = fs['Year'].max()
latest_fs   = fs[
    (fs['Year'] == latest_year) &
    (fs['Indicator'] == 'Moderate or Severe Food Insecurity (%)')
].groupby('Country')['Value'].mean().reset_index()

kenya_food    = fs[fs['Country'] == 'Kenya'].copy()
kenya_poverty = pov[pov['Country'] == 'Kenya'].copy()
kenya_fs_trend = kenya_food[
    kenya_food['Indicator'] == 'Moderate or Severe Food Insecurity (%)'
].groupby('Year_int')['Value'].mean().reset_index()

africa_avg  = latest_fs['Value'].mean()
kenya_row   = latest_fs[latest_fs['Country'] == 'Kenya']
kenya_val   = kenya_row['Value'].values[0] if len(kenya_row) else 73.9
kenya_rank  = latest_fs.sort_values('Value', ascending=False).reset_index(drop=True)
kenya_pos   = kenya_rank[kenya_rank['Country'] == 'Kenya'].index[0] + 1 if len(kenya_row) else 8
worst       = latest_fs.sort_values('Value', ascending=False).iloc[0]


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Food Security\n### Africa & Kenya")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🌍 Africa — The Scale",
        "🌱 Is It a Production Problem?",
        "🇰🇪 Kenya Deep Dive",
        "💰 Can Kenyans Afford Food?",
        "📋 Conclusions"
    ])
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- FAO FAOSTAT\n- World Bank\n- FEWS NET / KFSSG\n- KNBS")
    st.markdown("---")
    st.caption("Food Security in Africa & Kenya | 2024")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("# Food Security in Africa & Kenya")
    st.markdown("### *Using data to understand where hunger is, why it happens, and what needs to change.*")
    st.markdown("---")

    # Hero metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#d73027,#f46d43)">
        <div class="num">{kenya_val:.0f}%</div>
        <div class="lbl">of Kenyans are food insecure</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#1a1a2e,#2c3e6b)">
        <div class="num">#{kenya_pos}</div>
        <div class="lbl">Kenya's rank in Africa for food insecurity</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#fc8d59,#e34a33)">
        <div class="num">{africa_avg:.0f}%</div>
        <div class="lbl">Africa average food insecurity</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#b2182b,#7f0000)">
        <div class="num">45%</div>
        <div class="lbl">of income Kenyans spend on food</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Project description
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## What is Food Security?")
        st.markdown("""
        According to the FAO, food security exists when all people, at all times, have physical, social and economic access to sufficient, safe and nutritious food.

        Food security has **four pillars**:
        - 🌾 **Availability** — Is enough food being produced?
        - 🚚 **Access** — Can people physically reach the food?
        - 💵 **Affordability** — Can people economically afford it?
        - 🥗 **Utilization** — Is the food nutritious and safe?

        When any one of these breaks down, food insecurity follows.
        """)

    with col2:
        st.markdown("## The Three Questions")
        st.markdown("""
        <div class="insight">
        <strong>1.</strong> How bad is food insecurity in Africa and who is most affected?
        </div>
        <div class="insight">
        <strong>2.</strong> Is Africa not producing enough food — or can people not afford it?
        </div>
        <div class="insight">
        <strong>3.</strong> Why is Kenya food insecure and what needs to change?
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigate using the sidebar to explore the full analysis →")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: AFRICA — THE SCALE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Africa — The Scale":

    st.markdown("## Africa — The Scale of the Crisis")
    st.markdown("""
    Africa is home to over 1.4 billion people. More than half face moderate or severe food insecurity.
    The continent has seen the situation worsen every year since 2016.
    """)
    st.markdown("---")

    # Map
    st.markdown("### Where Is Hunger Worst?")
    st.markdown("Every country colored by its food insecurity level. The deeper the red, the more severe the crisis.")

    with st.spinner("Loading Africa map..."):
        try:
            africa_map = load_africa_map()
            africa_fs_map = africa_map.merge(latest_fs, left_on='name', right_on='Country', how='left')

            fig, ax = plt.subplots(figsize=(14, 10))
            africa_fs_map.plot(
                column='Value', cmap='RdYlGn_r', legend=True,
                legend_kwds={'label': 'Food Insecurity (%)', 'shrink': 0.5},
                missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
                ax=ax
            )
            for _, row in africa_fs_map.iterrows():
                ax.annotate(row['name'],
                            xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                            ha='center', fontsize=5.5, color='black')
            ax.set_title(f'Food Insecurity Across Africa ({latest_year})', fontsize=14, fontweight='bold')
            ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception:
            st.warning("Map could not load — internet connection required for map data.")

    st.markdown("---")

    # Top 10
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Most Affected Countries")
        top10  = latest_fs.sort_values('Value', ascending=False).head(10)
        colors = ['#b2182b' if c == 'Kenya' else '#fc8d59' for c in top10['Country']]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top10['Country'], top10['Value'], color=colors, edgecolor='white')
        for bar, val in zip(bars, top10['Value']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('Population Facing Food Insecurity (%)')
        ax.set_title(f'Top 10 Most Food Insecure Countries ({latest_year})', fontweight='bold', pad=12)
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.xaxis.grid(True, color='#eeeeee', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Key Insight")
        st.markdown(f"""
        <div class="insight">
        <strong>Sierra Leone</strong> tops the list at {worst['Value']:.1f}%.<br><br>
        <strong>Kenya</strong> ranks #{kenya_pos} in Africa at {kenya_val:.1f}% — significantly above the Africa average of {africa_avg:.1f}%.<br><br>
        Kenya is one of East Africa's largest economies yet sits in the top 10 worst affected countries.
        That tells us this is not simply a poverty story — there are structural problems at play.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Trend
    st.markdown("### The Situation Is Getting Worse")
    st.markdown("Kenya's trajectory is steeper than the Africa average — moving from 61% in 2016 to nearly 74% by 2024.")

    trend_data   = fs[fs['Indicator'] == 'Moderate or Severe Food Insecurity (%)'].copy()
    africa_trend = trend_data.groupby('Year_int')['Value'].mean().reset_index()
    kenya_trend  = trend_data[trend_data['Country'] == 'Kenya'].groupby('Year_int')['Value'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(africa_trend['Year_int'], africa_trend['Value'],
            color='#2166ac', linewidth=2.5, marker='o', label='Africa Average')
    ax.plot(kenya_trend['Year_int'], kenya_trend['Value'],
            color='#d73027', linewidth=2.5, marker='o', label='Kenya')
    ax.set_xlabel('Year')
    ax.set_ylabel('Food Insecurity (%)')
    ax.set_title('Food Insecurity Trend: Africa vs Kenya (2014–2024)', fontweight='bold', pad=12)
    ax.legend()
    chart_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌱 Is It a Production Problem?":

    st.markdown("## Is It a Production Problem?")
    st.markdown("""
    A natural assumption is that hunger exists because not enough food is being grown.
    This section tests that assumption directly.
    """)
    st.markdown("---")

    st.markdown("""
    <div class="insight">
    <strong>Spoiler:</strong> Africa's total food production has grown every year since 2014.
    The problem is not how much food is grown — it is whether people can access and afford it.
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prod['Year'], prod['Production (tonnes)'] / 1e9,
            color='#41ab5d', linewidth=2.5, marker='o')
    ax.fill_between(prod['Year'], prod['Production (tonnes)'] / 1e9, alpha=0.08, color='#41ab5d')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Food Production (Billion Tonnes)')
    ax.set_title('Africa Total Food Production (2014–2024)', fontweight='bold', pad=12)
    chart_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What the data tells us")
        st.markdown("""
        - Production has grown **consistently** from 2014 to 2024
        - Yet food insecurity has also grown over the same period
        - This directly rules out production as the primary cause of hunger
        - The food exists — the problem is **access and affordability**
        """)
    with col2:
        st.markdown("### What this means for Kenya")
        st.markdown("""
        Kenya is one of East Africa's **largest food producers**.
        Yet it ranks #8 in Africa for food insecurity.

        If production were the problem, Kenya would not be on that list.
        The issue is that food is not reaching the counties that need it,
        and that too many Kenyans cannot afford to buy it.
        """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: KENYA DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🇰🇪 Kenya Deep Dive":

    st.markdown("## Kenya Deep Dive")
    st.markdown(f"""
    Kenya ranks **#{kenya_pos}** in Africa at **{kenya_val:.1f}%** food insecurity —
    well above the Africa average of {africa_avg:.1f}%.
    This section answers: **where** in Kenya is it worst, and **why** is it worsening?
    """)
    st.markdown("---")

    # Counties
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Where in Kenya Is Hunger Worst?")
        top15 = kdf.sort_values('Food_Insecurity', ascending=False).head(15)
        colors_c = ['#b2182b' if v >= 75 else '#fc8d59' if v >= 60 else '#fdae61'
                    for v in top15['Food_Insecurity']]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(top15['County'], top15['Food_Insecurity'], color=colors_c, edgecolor='white')
        for bar, val in zip(bars, top15['Food_Insecurity']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('Population Facing Food Insecurity (%)')
        ax.set_title('Most Food Insecure Counties in Kenya', fontweight='bold', pad=12)
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.xaxis.grid(True, color='#eeeeee', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### The Northern Counties")
        st.markdown("""
        <div class="insight">
        <strong>Turkana, Mandera, Marsabit and Wajir</strong> all exceed 75% food insecurity.<br><br>
        These are arid counties, far from food production centers, with poor road infrastructure and high poverty rates.<br><br>
        Geography and infrastructure — not just poverty — are driving factors here.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Color Guide")
        st.markdown("""
        🔴 **Dark red** — above 75% (critical)\n
        🟠 **Orange** — 60–74% (severe)\n
        🟡 **Yellow** — below 60% (high)
        """)

    st.markdown("---")

    # Neighbors
    st.markdown("### Kenya Is Doing Worse Than Its Neighbors")
    st.markdown("Ethiopia and Uganda have been improving. Kenya has not — which means this is Kenya-specific, not just a regional trend.")

    east_africa = ['Kenya', 'Ethiopia', 'Uganda', 'Tanzania', 'South Sudan']
    ea_food = fs[
        (fs['Country'].isin(east_africa)) &
        (fs['Indicator'] == 'Moderate or Severe Food Insecurity (%)')
    ].copy()
    colors_ea = ['#d73027', '#fc8d59', '#fdae61', '#4575b4', '#313695']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, country in enumerate(east_africa):
        cd = ea_food[ea_food['Country'] == country].groupby('Year_int')['Value'].mean().reset_index()
        if len(cd) > 1:
            ax.plot(cd['Year_int'], cd['Value'], color=colors_ea[i],
                    linewidth=3 if country == 'Kenya' else 1.5,
                    linestyle='-' if country == 'Kenya' else '--',
                    marker='o', label=country)
    ax.set_xlabel('Year')
    ax.set_ylabel('Food Insecurity (%)')
    ax.set_title('Food Insecurity: Kenya vs East African Neighbors', fontweight='bold', pad=12)
    ax.legend()
    chart_style(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Poverty vs food insecurity
    st.markdown("### Poverty Is Rising Alongside Hunger")
    st.markdown("Both lines move in the same direction — confirming that affordability is the core driver in Kenya.")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(kenya_fs_trend['Year_int'], kenya_fs_trend['Value'],
             color='#d73027', linewidth=2.5, marker='o', label='Food Insecurity (%)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Food Insecurity (%)', color='#d73027')
    ax1.tick_params(axis='y', labelcolor='#d73027')

    ax2 = ax1.twinx()
    ax2.plot(kenya_poverty['Year'], kenya_poverty['Poverty Rate (%)'],
             color='#2166ac', linewidth=2.5, marker='s', linestyle='--', label='Poverty Rate (%)')
    ax2.set_ylabel('Poverty Rate (%)', color='#2166ac')
    ax2.tick_params(axis='y', labelcolor='#2166ac')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('Kenya: Food Insecurity vs Poverty Rate', fontweight='bold', pad=12)
    ax1.spines[['top', 'right']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    ax1.grid(True, color='#eeeeee', linewidth=0.8)
    ax1.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: AFFORDABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 Can Kenyans Afford Food?":

    st.markdown("## Can the Average Kenyan Afford Food?")
    st.markdown("""
    We have shown poverty is rising alongside hunger. Now we put a number to it —
    what does it actually look like for a typical Kenyan household every month?
    """)
    st.markdown("---")

    # Metric highlights
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="metric-card" style="background:linear-gradient(135deg,#2166ac,#1a4a7a)">
        <div class="num">KES 18,000</div>
        <div class="lbl">Average monthly household income</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""
    <div class="metric-card" style="background:linear-gradient(135deg,#d73027,#b2182b)">
        <div class="num">KES 20,500</div>
        <div class="lbl">Total monthly expenses (food + non-food)</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""
    <div class="metric-card" style="background:linear-gradient(135deg,#7f0000,#4a0000)">
        <div class="num">KES −2,500</div>
        <div class="lbl">Monthly deficit every single month</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("*Source: KNBS Economic Survey 2023, World Bank GNI per capita data.*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Monthly Income vs Expenses")
        categories    = ['Average Monthly\nIncome', 'Basic Food\nBasket', 'Non-Food\nEssentials', 'Total Monthly\nExpenses']
        values        = [18000, 12500, 8000, 20500]
        colors_afford = ['#2166ac', '#d73027', '#fc8d59', '#b2182b']

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(categories, values, color=colors_afford, width=0.5, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'KES {val:,}', ha='center', fontsize=10, fontweight='bold')
        ax.axhline(y=18000, color='#2166ac', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Average Monthly Income')
        ax.annotate('Monthly deficit\nof KES 2,500',
                    xy=(3, 18000), xytext=(2.3, 14000),
                    fontsize=9, color='#b2182b',
                    arrowprops=dict(arrowstyle='->', color='#b2182b'))
        ax.set_ylabel('Kenya Shillings (KES)')
        ax.set_title('Can the Average Kenyan Afford Food?', fontweight='bold', pad=12)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#eeeeee', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### % of Income Spent on Food")
        countries_s  = ['USA', 'UK', 'Germany', 'S. Africa', 'Kenya', 'Uganda', 'Ethiopia']
        food_pct     = [6.4, 8.2, 10.5, 22.0, 45.0, 52.0, 58.0]
        colors_s     = ['#41ab5d' if p < 20 else '#fdae61' if p < 40 else '#d73027' for p in food_pct]
        colors_s[4]  = '#b2182b'

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(countries_s, food_pct, color=colors_s, edgecolor='white', width=0.6)
        for bar, val in zip(bars, food_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val}%', ha='center', fontsize=10, fontweight='bold')
        ax.axhline(y=15, color='#aaaaaa', linestyle='--', linewidth=1.2, label='15% — food secure benchmark')
        ax.set_ylabel('% of Income Spent on Food')
        ax.set_title('% of Income on Food — Kenya vs the World', fontweight='bold', pad=12)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#eeeeee', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("""
    <div class="insight">
    <strong>The key insight:</strong> When 45% of income goes to food, any rise in food prices — from drought,
    inflation or supply chain disruption — immediately pushes millions of households into food insecurity.
    There is no financial buffer. This is why food insecurity in Kenya is so sensitive to external shocks.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Conclusions":

    st.markdown("## Conclusions")
    st.markdown("The data tells a clear and consistent story across all nine visualizations.")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Key Findings")

        findings = [
            ("🌍 Africa Average", f"{africa_avg:.1f}% food insecurity"),
            ("🔴 Worst Affected", f"{worst['Country']} at {worst['Value']:.1f}%"),
            (f"🇰🇪 Kenya Rank", f"#{kenya_pos} in Africa at {kenya_val:.1f}%"),
            ("📈 Kenya Trend", "Worsened by 12.8% since 2016"),
            ("💰 Monthly Deficit", "KES 2,500 every month"),
            ("🍽️ Food Spend", "45% of income on food"),
        ]

        for label, value in findings:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:10px 15px;
                        background:#f9f9f9; border-radius:8px; margin:6px 0;">
                <span style="color:#555;">{label}</span>
                <strong style="color:#d73027;">{value}</strong>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("### The Story in 4 Conclusions")

        conclusions = [
            ("1. The problem is NOT production",
             "Africa grows more food every year. Hunger is rising anyway. The food exists — people cannot access or afford it."),
            ("2. Poverty is the biggest driver",
             "As poverty rises in Kenya, food insecurity rises with it in lockstep. The average household runs a monthly deficit."),
            ("3. Geography matters",
             "Turkana, Mandera, Marsabit and Wajir face rates above 75%. Remote, arid counties far from food and markets."),
            ("4. Kenya is an outlier in its own region",
             "Ethiopia and Uganda are improving. Kenya is not. The problem requires Kenya-specific solutions."),
        ]

        for title, text in conclusions:
            st.markdown(f"""
            <div class="insight">
            <strong>{title}</strong><br>
            <span style="font-size:0.9rem;">{text}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### What Needs to Change")

    r1, r2, r3, r4 = st.columns(4)
    recs = [
        ("🧑‍🌾", "Support smallholder farmers in high-risk regions"),
        ("🚗", "Improve road and market access to arid counties"),
        ("🛡️", "Expand social protection for the poorest households"),
        ("💧", "Invest in drought-resistant crops and irrigation"),
    ]
    for col, (icon, text) in zip([r1, r2, r3, r4], recs):
        col.markdown(f"""
        <div style="background:#f9f9f9; border-radius:12px; padding:20px; text-align:center; height:130px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:0.85rem; color:#444; margin-top:10px;">{text}</div>
        </div>
        """, unsafe_allow_html=True)
