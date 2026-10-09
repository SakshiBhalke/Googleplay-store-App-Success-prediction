import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Play Store Success Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    meta = joblib.load("meta.joblib")

    try:
        class_stats = joblib.load("class_stats.joblib")
    except FileNotFoundError:
        class_stats = None

    return model, scaler, encoders, meta, class_stats


model, scaler, encoders, meta, class_stats = load_artifacts()


# ============================================================
# FEATURES
# ============================================================

cat_cols = meta["cat_cols"]
num_cols = meta["num_cols"]


# ============================================================
# HEADER
# ============================================================

st.title("📱 Play Store Success Predictor")

st.subheader(
    "Machine Learning Dashboard for App Popularity"
)

st.write(
    "Predict whether an application belongs to the "
    "1,000,000+ installs group using its characteristics."
)

st.info(
    f"🤖 **Model:** {meta['best_model_name']}   |   "
    "🎯 **Target:** 1,000,000+ installs"
)


st.divider()


# ============================================================
# DASHBOARD INFORMATION
# ============================================================

st.subheader("📊 Dashboard Overview")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        label="🎯 Prediction Target",
        value="1M+ Installs"
    )

with info2:
    st.metric(
        label="🤖 ML Model",
        value=meta["best_model_name"]
    )

with info3:
    st.metric(
        label="📋 Input Features",
        value="8"
    )

with info4:
    st.metric(
        label="📈 Visualization",
        value="Enabled"
    )


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📋 Enter Application Details")

st.write(
    "Provide the following information about the app."
)


with st.form("prediction_form"):

    left_col, right_col = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left_col:

        st.subheader("📱 App Information")

        category = st.selectbox(
            "📂 App Category",
            sorted(
                encoders["Category"].classes_
            )
        )

        content_rating = st.selectbox(
            "🔞 Content Rating",
            sorted(
                encoders["Content Rating"].classes_
            )
        )

        app_type = st.selectbox(
            "📱 App Type",
            sorted(
                encoders["Type"].classes_
            )
        )

        rating = st.slider(
            "⭐ Current Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.2,
            step=0.1
        )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right_col:

        st.subheader("📊 App Statistics")

        reviews = st.number_input(
            "💬 Number of Reviews",
            min_value=0,
            value=1000,
            step=100
        )

        size_mb = st.number_input(
            "📦 App Size (MB)",
            min_value=0.1,
            value=15.0,
            step=0.5
        )

        price = st.number_input(
            "💰 Price (USD)",
            min_value=0.0,
            value=0.0,
            step=0.5
        )

        days_since_update = st.number_input(
            "🕒 Days Since Last Update",
            min_value=0,
            value=90,
            step=10
        )


    st.divider()


    submitted = st.form_submit_button(
        "🚀 PREDICT APP POPULARITY",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # ========================================================
    # CREATE INPUT
    # ========================================================

    row = {

        "Category":
            encoders["Category"].transform(
                [category]
            )[0],

        "Type":
            encoders["Type"].transform(
                [app_type]
            )[0],

        "Content Rating":
            encoders["Content Rating"].transform(
                [content_rating]
            )[0],

        "Rating":
            rating,

        "Reviews":
            reviews,

        "Size_MB":
            size_mb,

        "Price":
            price,

        "Days_Since_Update":
            days_since_update
    }


    # ========================================================
    # DATAFRAME
    # ========================================================

    X_input = pd.DataFrame(
        [row]
    )[
        cat_cols + num_cols
    ]


    # ========================================================
    # SCALE NUMERICAL FEATURES
    # ========================================================

    X_input_scaled = X_input.copy()

    X_input_scaled[num_cols] = scaler.transform(
        X_input[num_cols]
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        X_input_scaled
    )[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("🔮 Prediction Result")


    if prediction == 1:

        st.success(
            "🚀 LIKELY POPULAR\n\n"
            "The model predicts that this app belongs "
            "to the 1,000,000+ installs group."
        )

    else:

        st.warning(
            "📉 LOWER INSTALL GROUP\n\n"
            "The model predicts that this app belongs "
            "to the fewer-than-1,000,000-installs group."
        )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.subheader("📌 Your App Metrics")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "💬 Reviews",
            f"{reviews:,}"
        )

    with m2:
        st.metric(
            "⭐ Rating",
            f"{rating:.1f} / 5"
        )

    with m3:
        st.metric(
            "📦 App Size",
            f"{size_mb:g} MB"
        )

    with m4:
        st.metric(
            "💰 Price",
            f"${price:g}"
        )


    # ========================================================
    # APP PROFILE CHART
    # ========================================================

    st.divider()

    st.header("📊 Your App Profile")

    st.write(
        "Visual summary of the numerical features entered above."
    )


    profile_labels = [
        "Rating",
        "Reviews",
        "App Size",
        "Price",
        "Days Since Update"
    ]


    # Normalize values only for visualization
    profile_values = [

        (rating / 5) * 100,

        min(
            (reviews / 10000) * 100,
            100
        ),

        min(
            (size_mb / 100) * 100,
            100
        ),

        min(
            (price / 100) * 100,
            100
        ),

        min(
            (days_since_update / 365) * 100,
            100
        )
    ]


    profile_text = [

        f"{rating:.1f}/5",

        f"{reviews:,}",

        f"{size_mb:g} MB",

        f"${price:g}",

        f"{days_since_update} days"
    ]


    profile_chart = go.Figure()


    profile_chart.add_trace(
        go.Bar(
            x=profile_labels,
            y=profile_values,
            text=profile_text,
            textposition="outside",
            marker_color=[
                "#6366F1",
                "#3B82F6",
                "#06B6D4",
                "#8B5CF6",
                "#F59E0B"
            ]
        )
    )


    profile_chart.update_layout(
        title="App Feature Overview",
        height=450,
        yaxis=dict(
            title="Normalized Value",
            range=[0, 110]
        ),
        xaxis=dict(
            title="Features"
        ),
        template="plotly_white",
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50
        )
    )


    st.plotly_chart(
        profile_chart,
        use_container_width=True
    )


    # ========================================================
    # RATING GAUGE
    # ========================================================

    st.divider()

    st.header("⭐ Rating Analysis")


    rating_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rating,
            number={
                "suffix": " / 5"
            },
            title={
                "text": "Current App Rating"
            },
            gauge={
                "axis": {
                    "range": [1, 5]
                },

                "bar": {
                    "color": "#4F46E5"
                },

                "steps": [

                    {
                        "range": [1, 2.5],
                        "color": "#FEE2E2"
                    },

                    {
                        "range": [2.5, 3.5],
                        "color": "#FEF3C7"
                    },

                    {
                        "range": [3.5, 4.5],
                        "color": "#DBEAFE"
                    },

                    {
                        "range": [4.5, 5],
                        "color": "#DCFCE7"
                    }
                ]
            }
        )
    )


    rating_gauge.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(
            l=40,
            r=40,
            t=60,
            b=30
        )
    )


    st.plotly_chart(
        rating_gauge,
        use_container_width=True
    )


    # ========================================================
    # DATASET COMPARISON
    # ========================================================

    if class_stats is not None:

        st.divider()

        st.header("📊 Dataset Comparison")

        st.write(
            "Compare the current app with average values "
            "from the two groups in the dataset."
        )


        labels = [
            "Rating ⭐",
            "Size (MB)"
        ]


        this_values = [
            rating,
            size_mb
        ]


        lower_group_values = [
            class_stats.loc[0, "Rating"],
            class_stats.loc[0, "Size_MB"]
        ]


        higher_group_values = [
            class_stats.loc[1, "Rating"],
            class_stats.loc[1, "Size_MB"]
        ]


        comparison_chart = go.Figure()


        comparison_chart.add_trace(
            go.Bar(
                name="Fewer Than 1M Installs",
                x=labels,
                y=lower_group_values,
                marker_color="#94A3B8"
            )
        )


        comparison_chart.add_trace(
            go.Bar(
                name="1M+ Installs",
                x=labels,
                y=higher_group_values,
                marker_color="#6366F1"
            )
        )


        comparison_chart.add_trace(
            go.Bar(
                name="Your App",
                x=labels,
                y=this_values,
                marker_color="#22C55E"
            )
        )


        comparison_chart.update_layout(
            barmode="group",
            height=450,
            template="plotly_white",
            title="Rating and Size Comparison",
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=60
            )
        )


        st.plotly_chart(
            comparison_chart,
            use_container_width=True
        )


        # ====================================================
        # REVIEWS COMPARISON
        # ====================================================

        st.header("💬 Reviews Comparison")

        st.write(
            "Review counts are displayed using a logarithmic scale "
            "because review counts can vary significantly."
        )


        review_labels = [
            "Fewer Than 1M",
            "1M+ Installs",
            "Your App"
        ]


        review_values = [

            class_stats.loc[
                0,
                "Reviews"
            ],

            class_stats.loc[
                1,
                "Reviews"
            ],

            reviews
        ]


        review_chart = go.Figure()


        review_chart.add_trace(
            go.Bar(
                x=review_labels,
                y=review_values,
                text=[
                    f"{value:,.0f}"
                    for value in review_values
                ],
                textposition="outside",
                marker_color=[
                    "#94A3B8",
                    "#6366F1",
                    "#22C55E"
                ]
            )
        )


        review_chart.update_layout(
            height=450,
            template="plotly_white",
            title="Reviews Comparison",
            yaxis=dict(
                type="log",
                title="Reviews (Log Scale)"
            ),
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=50
            )
        )


        st.plotly_chart(
            review_chart,
            use_container_width=True
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if hasattr(
        model,
        "feature_importances_"
    ):

        st.divider()

        st.header("📈 Feature Importance")

        st.write(
            "Relative importance of the features used by "
            "the machine-learning model."
        )


        importance = pd.Series(
            model.feature_importances_,
            index=cat_cols + num_cols
        ).sort_values()


        importance_chart = go.Figure()


        importance_chart.add_trace(
            go.Bar(
                x=importance.values,
                y=importance.index,
                orientation="h",
                marker_color="#6366F1"
            )
        )


        importance_chart.update_layout(
            height=450,
            template="plotly_white",
            title="Model Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=50
            )
        )


        st.plotly_chart(
            importance_chart,
            use_container_width=True
        )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    st.divider()

    with st.expander(
        "🔍 View Data Sent to Model"
    ):

        st.dataframe(
            X_input,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.header("ℹ️ About This Project")

st.write(
    "This application uses Machine Learning to estimate "
    "whether a Google Play Store app belongs to the "
    "1,000,000+ installs group."
)


st.subheader("📋 Features Used")

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.write("📂 Category")
    st.write("📱 Type")
    st.write("🔞 Content Rating")
    st.write("⭐ Rating")


with about_col2:

    st.write("💬 Reviews")
    st.write("📦 App Size")
    st.write("💰 Price")
    st.write("🕒 Days Since Last Update")


st.warning(
    "⚠️ This is a machine-learning estimate based on "
    "the training data. It is not a guarantee of future "
    "app installs."
)