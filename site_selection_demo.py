import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import datetime
import random
import time
import numpy as np
import plotly.express as px
from plotly import graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import skewnorm
import pydeck as pdk
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
import geopandas as gpd
import leafmap.colormaps as cm
from leafmap.common import hex_to_rgb
from streamlit_extras.switch_page_button import switch_page
import pyautogui

# STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
# # We create a downloads directory within the streamlit static asset directory
# # and we write output files to it
# DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
# if not DOWNLOADS_PATH.is_dir():
#     DOWNLOADS_PATH.mkdir()

# DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
# if not DOWNLOADS_PATH.is_dir():
#     DOWNLOADS_PATH.mkdir()

link_prefix = "https://raw.githubusercontent.com/giswqs/data/main/housing/"

random.seed(42)

data_links = {

    "monthly_current": {
        "state": link_prefix + "Core/RDC_Inventory_Core_Metrics_State.csv",
        "county": link_prefix + "Core/RDC_Inventory_Core_Metrics_County.csv",
    },
}

def get_data_columns(df, category, frequency="monthly"):
    if frequency == "monthly":
        if category.lower() == "county":
            del_cols = ["month_date_yyyymm", "county_fips", "county_name"]
        elif category.lower() == "state":
            del_cols = ["month_date_yyyymm", "state", "state_id", "STUSPS"]
    cols = df.columns.values.tolist()
    for col in cols:
        if col.strip() in del_cols:
            cols.remove(col)
    if category.lower() == "county":
        return cols[1:]
    elif category.lower() == "state":
        return cols[2:]
    
def get_inventory_data(url):
    df = pd.read_csv(url)
    url = url.lower()
    if "county" in url:
        df["county_fips"] = df["county_fips"].map(str)
        df["county_fips"] = df["county_fips"].str.zfill(5)
    elif "state" in url:
        df["STUSPS"] = df["state_id"].str.upper()
    elif "metro" in url:
        df["cbsa_code"] = df["cbsa_code"].map(str)
    elif "zip" in url:
        df["postal_code"] = df["postal_code"].map(str)
        df["postal_code"] = df["postal_code"].str.zfill(5)

    if "listing_weekly_core_aggregate_by_country" in url:
        columns = get_data_columns(df, "national", "weekly")
        for column in columns:
            if column != "median_days_on_market_by_day_yy":
                df[column] = df[column].str.rstrip("%").astype(float) / 100
    if "listing_weekly_core_aggregate_by_metro" in url:
        columns = get_data_columns(df, "metro", "weekly")
        for column in columns:
            if column != "median_days_on_market_by_day_yy":
                df[column] = df[column].str.rstrip("%").astype(float) / 100
        df["cbsa_code"] = df["cbsa_code"].str[:5]
    return df

def filter_weekly_inventory(df, week):
    df = df[df["week_end_date"] == week]
    return df


def get_start_end_year(df):
    start_year = int(str(df["month_date_yyyymm"].min())[:4])
    end_year = int(str(df["month_date_yyyymm"].max())[:4])
    return start_year, end_year


def get_periods(df):
    return [str(d) for d in list(set(df["month_date_yyyymm"].tolist()))]

@st.cache_data
def get_geom_data(category):

    prefix = (
        "https://raw.githubusercontent.com/giswqs/streamlit-geospatial/master/data/"
    )
    links = {
        "national": prefix + "us_nation.geojson",
        "state": prefix + "us_states.geojson",
        "county": prefix + "us_counties.geojson",
        "metro": prefix + "us_metro_areas.geojson",
        "zip": "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_zcta510_500k.zip",
    }

    if category.lower() == "zip":
        r = requests.get(links[category])
        out_zip = os.path.join(DOWNLOADS_PATH, "cb_2018_us_zcta510_500k.zip")
        with open(out_zip, "wb") as code:
            code.write(r.content)
        zip_ref = zipfile.ZipFile(out_zip, "r")
        zip_ref.extractall(DOWNLOADS_PATH)
        gdf = gpd.read_file(out_zip.replace("zip", "shp"))
    else:
        gdf = gpd.read_file(links[category])
    return gdf


def join_attributes(gdf, df, category):

    new_gdf = None
    if category == "county":
        new_gdf = gdf.merge(df, left_on="GEOID", right_on="county_fips", how="outer")
    elif category == "state":
        new_gdf = gdf.merge(df, left_on="STUSPS", right_on="STUSPS", how="outer")
    return new_gdf


def select_non_null(gdf, col_name):
    new_gdf = gdf[~gdf[col_name].isna()]
    return new_gdf


def select_null(gdf, col_name):
    new_gdf = gdf[gdf[col_name].isna()]
    return new_gdf


def get_data_dict(name):
    in_csv = os.path.join(os.getcwd(), "data/realtor_data_dict.csv")
    df = pd.read_csv(in_csv)
    label = list(df[df["Name"] == name]["Label"])[0]
    desc = list(df[df["Name"] == name]["Description"])[0]
    return label, desc


def get_weeks(df):
    seq = list(set(df[~df["week_end_date"].isnull()]["week_end_date"].tolist()))
    weeks = [
        datetime.date(int(d.split("/")[2]), int(d.split("/")[0]), int(d.split("/")[1]))
        for d in seq
    ]
    weeks.sort()
    return weeks


def get_saturday(in_date):
    idx = (in_date.weekday() + 1) % 7
    sat = in_date + datetime.timedelta(6 - idx)
    return sat

def get_color(score):
    if score < 20:
        return 'red'
    elif score < 40:
        return 'orange'
    elif score < 60:
        return 'yellow'
    elif score < 80:
        return 'green'
    else:
        return 'blue'

 
def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)

# if 'count' not in st.session_state:
# 	st.session_state.count = 0
        



def app():
    # perhaps use plotly
    #print(st.session_state.count)
    st.set_page_config(
        page_title="Granger + SVN Demo", layout="wide"
    )
    css = r'''
        <style>
            [data-testid="stForm"] {border: 0px}
        </style>
    '''
    st.markdown(css, unsafe_allow_html=True)
    st.title("Granger/SVN")
    st.header("Site Selection Demo")
    st.write('')

    TABS = ["Map", "Details"]
    MAP_TAB, DETAILS_TAB = st.tabs(TABS)

    with MAP_TAB:

        row1_col1, row1_col2 = st.columns(
            [0.6, 0.8]
        )

        frequency = "Monthly"

        with row1_col1:
                scale = st.selectbox(
                    "Scale", ["State", "County"], index=1
                )

        gdf = get_geom_data(scale.lower())

        inventory_df = get_inventory_data(
                data_links["monthly_current"][scale.lower()]
            )
        
        # added random values
        if scale.lower() == "state":
            b = inventory_df["STUSPS"]
            inventory_df = inventory_df.iloc[:, :3]
            inventory_df["STUSPS"] = b
        else:
            inventory_df = inventory_df.iloc[:, :3]
        
        inventory_df['IRR'] = np.random.randint(-10, 55, inventory_df.shape[0])
        inventory_df['Demographic Score'] = np.random.randint(0, 100, inventory_df.shape[0])
        inventory_df['Location Score'] = np.random.randint(0, 100, inventory_df.shape[0])
        inventory_df['Financial Score'] = np.random.randint(0, 100, inventory_df.shape[0])
        inventory_df['Overall Score'] = (inventory_df["Demographic Score"]+inventory_df["Financial Score"]+inventory_df["Location Score"])/3
        inventory_df["Average Price Per Square Foot"] = random.randrange(30, 240)

        selected_period = get_periods(inventory_df)[0]

        data_cols = get_data_columns(inventory_df, scale.lower(), frequency.lower())

        with row1_col2:
            selected_col = st.selectbox("Attribute", data_cols)


        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(
            [0.6, 0.68, 0.7, 0.7]
        )

        palettes = cm.list_colormaps()
        # with row2_col1:
        palette = "RdYlGn"
        #palette = st.selectbox("Color palette", palettes, index=palettes.index("Blues"))
        # with row2_col2:
        #     n_colors = st.slider("Number of colors", min_value=2, max_value=20, value=8)
        with row2_col1:
            show_nodata = st.checkbox("Show nodata areas", value=True)
        with row2_col2:
            show_3d = st.checkbox("Show 3D view", value=False)
        with row2_col3:
            if show_3d:
                elev_scale = st.slider(
                    "Elevation scale", min_value=1, max_value=10000, value=1, step=10
                )
                with row2_col4:
                    st.info("Press Ctrl and move the left mouse button.")
            else:
                elev_scale = 1

        gdf = join_attributes(gdf, inventory_df, scale.lower())
        gdf_null = select_null(gdf, selected_col)
        gdf = select_non_null(gdf, selected_col)
        gdf = gdf.sort_values(by=selected_col, ascending=True)

        colors = cm.get_palette(palette, 8)
        colors = [hex_to_rgb(c) for c in colors]

        for i, ind in enumerate(gdf.index):
            index = int(i / (len(gdf) / len(colors)))
            if index >= len(colors):
                index = len(colors) - 1
            gdf.loc[ind, "R"] = colors[index][0]
            gdf.loc[ind, "G"] = colors[index][1]
            gdf.loc[ind, "B"] = colors[index][2]

        initial_view_state = pdk.ViewState(
            latitude=40,
            longitude=-100,
            zoom=3,
            max_zoom=16,
            pitch=0,
            bearing=0,
            height=900,
            width=None,
        )

        min_value = gdf[selected_col].min()
        max_value = gdf[selected_col].max()
        color = "color"
        # color_exp = f"[({selected_col}-{min_value})/({max_value}-{min_value})*255, 0, 0]"
        color_exp = f"[R, G, B]"

        def switch():
            print("dupa")
            # st.session_state.count +=1
            # if st.session_state.count==2:
            #     switch_page("test")
        geojson = pdk.Layer(
            "GeoJsonLayer",
            gdf,
            pickable=True,
            opacity=0.5,
            stroked=True,
            filled=True,
            extruded=show_3d,
            wireframe=True,
            get_elevation=f"{selected_col}",
            elevation_scale=elev_scale,
            # get_fill_color="color",
            get_fill_color=color_exp,
            get_line_color=[0, 0, 0],
            get_line_width=2,
            line_width_min_pixels=1,
            on_click=print("dupa")
        )

        #print(st.session_state.count)

        geojson_null = pdk.Layer(
            "GeoJsonLayer",
            gdf_null,
            pickable=True,
            opacity=0.2,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_fill_color=[200, 200, 200],
            get_line_color=[0, 0, 0],
            get_line_width=2,
            line_width_min_pixels=1,
        )

        tooltip = {
            "html": "<b>Name:</b> {NAME}<br><b>Value:</b> {"
            + selected_col
            + "}<br><b>Date:</b> "
            + selected_period
            + "",
            "style": {"backgroundColor": "steelblue", "color": "white"},
        }

        layers = [geojson]
        if show_nodata:
            layers.append(geojson_null)

        r = pdk.Deck(
            layers=layers,
            initial_view_state=initial_view_state,
            map_style="light",
            tooltip=tooltip,
        )

        row3_col1, row3_col2 = st.columns([6, 1])

        with row3_col1:
            st.pydeck_chart(r)
        with row3_col2:
            st.write(
                cm.create_colormap(
                    palette,
                    label=selected_col.replace("_", " ").title(),
                    width=0.2,
                    height=3,
                    orientation="vertical",
                    vmin=min_value,
                    vmax=max_value,
                    font_size=10,
                )
            )
        row4_col1, row4_col2, row4_col3 = st.columns([1, 2, 3])
        with row4_col1:
            show_data = st.checkbox("Show raw data")
        with row4_col2:
            show_cols = st.multiselect("Select columns", data_cols)
        if show_data:
            if scale == "State":
                st.dataframe(gdf[["NAME", "STUSPS"] + show_cols])
            elif scale == "County":
                st.dataframe(gdf[["NAME", "STATEFP", "COUNTYFP"] + show_cols])


    with DETAILS_TAB:

        orig_data_df = pd.read_excel("granger_data.xlsx", sheet_name=0)
        results_df = pd.read_excel("granger_data.xlsx", sheet_name=1)

        st.dataframe(orig_data_df.iloc[:1,:])

        zip_code = orig_data_df.iloc[0]["ZIP code"]
        data_df = orig_data_df[orig_data_df["ZIP code"] != zip_code]

        expander = st.expander("Add new site")

        results_row = results_df[results_df["ZIP code"] == 36109].iloc[0]

        stats, irr = st.columns([3, 2])
        fin, dem, loc = stats.columns(3)
        with fin:
            st.header("Financial score")
            score = results_row['Financial Score']
            colour = get_color(score)
            st.markdown(f"# :{colour}[{score}%]")
        with dem:
            st.header("Demographic score")
            score = results_row['Demographic Score']
            colour = get_color(score)
            st.markdown(f"# :{colour}[{score}%]")
        with loc:
            st.header("Location score")
            score = results_row['Location Score']
            colour = get_color(score)
            st.markdown(f"# :{colour}[{score}%]")

        stats.markdown("""---""")

        with stats.columns([1, 2, 1])[1]:
            st.write("# Overall Score")
            score = results_row['Overall Score']
            colour = get_color(score)
            st.markdown(f"# :{colour}[{score:.0f}%]")

        stats.markdown("""---""")
        positive, negative = stats.columns(2)

        with positive:
            st.header("Positive factors")
            st.write(f"### {results_row['positive']}")
        with negative:
            st.header("Negative factors")
            st.write(f"### {results_row['negative']}")

        with irr:
            st.header("Estimated IRR")
            a = 4
            avg_irr = results_row["IRR"]
            r = skewnorm.rvs(a, size=1000, loc=avg_irr, scale=0.1 * avg_irr)
            fig = ff.create_distplot([r], ["IRR"],
                                    show_curve=True,
                                    show_rug=False,
                                    show_hist=False,
                                    histnorm='probability', bin_size=1)
            avg_irr = np.median(r)
            fig.add_trace(go.Scatter(x=[avg_irr, avg_irr], y=[0, fig.data[0].y.max()],
                                    mode='lines', name='Median IRR',
                                    line=dict(color='red', dash="dash")))
            st.plotly_chart(fig)
            st.write("### Median estimated IRR is {:.2f}%".format(avg_irr))

if __name__=="__main__":
    app()