import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 🔐 PASSWORD PROTECTION
PASSWORD = "geonly123"
pwd = st.text_input("Enter password", type="password")
if pwd != PASSWORD:
    st.warning("Access denied.")
    st.stop()

# 📁 FILE UPLOADER
st.title("SKU Matching Tool")
uploaded_file = st.file_uploader("Upload your SKU Excel file", type=["xlsx", "xls"])
if not uploaded_file:
    st.stop()

# 🧼 Load and preprocess Excel file
df_raw = pd.read_excel(uploaded_file, header=None)
df = df_raw.T
df.columns = df.iloc[0]
df = df[1:]
df.reset_index(drop=True, inplace=True)

# 🔍 Detect Description Row
keywords = ["cu ft", "side by side", "french door", '"', "in.", "top freezer", "bottom freezer", "refrigerator"]
searchable = df.applymap(lambda x: str(x).lower())
row_scores = []
for i, row in searchable.iterrows():
    score = sum(any(kw in cell for kw in keywords) for cell in row)
    if score > 2:
        row_scores.append((i, score))

if row_scores:
    best_row_index = sorted(row_scores, key=lambda x: x[1], reverse=True)[0][0]
    description_row = df.iloc[best_row_index]
    df_raw_rebuilt = pd.concat([
        df_raw.iloc[:2],
        pd.DataFrame([["Description"] + description_row.tolist()]),
        df_raw.iloc[2:]
    ]).reset_index(drop=True)
    df = df_raw_rebuilt.T
    df.columns = df.iloc[0]
    df = df[1:]
    df.reset_index(drop=True, inplace=True)

# 🔍 Basic cleaning
if 'SKU' not in df.columns:
    df.rename(columns={df.columns[0]: 'SKU'}, inplace=True)
df.fillna('', inplace=True)
df['SKU'] = df['SKU'].astype(str)

# 🤔 User Input
input_sku = st.text_input("Enter a SKU:")
search_type = st.selectbox("What kind of match do you want?", ["GE only", "Competitor (non-GE)"])
strict_config = st.checkbox("Strict configuration match", value=True)

# 🎛️ Feature Matching Preferences
st.subheader("🎛️ Feature Matching Preferences")
detected_features = [col for col in df.columns if col not in ['SKU', 'combined_specs']]
valid_features = []
if input_sku in df['SKU'].values:
    input_row = df[df['SKU'] == input_sku].iloc[0]
    valid_features = [col for col in detected_features if str(input_row[col]).strip() not in ['', 'nan', 'NaN']]
selected_features = st.multiselect("Which features are most important to match? Selected features will apear in the results table.", options=valid_features)

# 🛠️ Construct weighted spec string
df['combined_specs'] = ""
for col in selected_features:
    weight = 3
    if col in df.columns:
        df['combined_specs'] += ((df[col].astype(str) + " ") * weight)
if not selected_features:
    spec_columns = [col for col in df.columns if col not in ['SKU', 'combined_specs']]
    df['combined_specs'] = df[spec_columns].astype(str).agg(' '.join, axis=1)

# 🔢 TF-IDF Model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_specs'])

# 🔎 Matching Function
def find_matches(input_sku, brand_filter='ge', top_n=5, strict=True):
    input_row = df[df['SKU'] == input_sku]
    if input_row.empty:
        return f"SKU {input_sku} not found."

    input_index = input_row.index[0]
    similarities = cosine_similarity(tfidf_matrix[input_index], tfidf_matrix)[0]

    brand_col = 'Brand' if 'Brand' in df.columns else 'spec_14'
    config_col = 'Configuration' if 'Configuration' in df.columns else 'spec_7'
    status_col = 'Model Status' if 'Model Status' in df.columns else 'spec_9'
    description_col = 'Description' if 'Description' in df.columns else None

    input_config = input_row.iloc[0][config_col]
    df_copy = df.copy()
    df_copy['similarity'] = similarities

    if strict:
        same_config = df_copy[config_col].str.lower() == input_config.lower()
    else:
        same_config = df_copy[config_col].notna()

    if brand_filter == "ge":
        filtered = df_copy[
            (df_copy[brand_col].str.lower() == 'ge') &
            same_config &
            (df_copy[status_col].str.lower() == 'active model') &
            (df_copy['SKU'] != input_sku)
        ]
    else:
        filtered = df_copy[
            (df_copy[brand_col].str.lower() != 'ge') &
            same_config &
            (df_copy[status_col].str.lower() == 'active model') &
            (df_copy['SKU'] != input_sku)
        ]

    filtered = filtered.sort_values(by='similarity', ascending=False)

    columns_to_return = ['SKU', brand_col]
    if description_col:
        columns_to_return.append(description_col)
    columns_to_return += [config_col, status_col]
    for feat in selected_features:
        if feat not in columns_to_return:
            columns_to_return.append(feat)

    rename_dict = {
        brand_col: 'Brand',
        config_col: 'Configuration',
        status_col: 'Model Status',
    }
    if description_col:
        rename_dict[description_col] = 'Description'

    return filtered[columns_to_return].rename(columns=rename_dict).head(top_n)

# 🖥️ Show Matches
if input_sku:
    # Brand detection display
    sku_row = df[df['SKU'] == input_sku]
    if not sku_row.empty and 'Brand' in df.columns:
        entered_brand = sku_row.iloc[0]['Brand']
        st.info(f"Entered SKU appears to be a `{entered_brand}` product.")

    competitor_row = sku_row
    if not competitor_row.empty:
        brand_col = 'Brand'
        config_col = 'Configuration'
        status_col = 'Model Status'
        description_col = 'Description'

        competitor_data = {
            "SKU": input_sku,
            "Brand": competitor_row.iloc[0].get(brand_col, ''),
            "Configuration": competitor_row.iloc[0].get(config_col, ''),
            "Model Status": competitor_row.iloc[0].get(status_col, '')
        }
        if description_col:
            competitor_data["Description"] = str(competitor_row[description_col].values[0]).strip()
        for feat in selected_features:
            competitor_data[feat] = competitor_row.iloc[0].get(feat, '')
        competitor_df = pd.DataFrame([competitor_data])
        st.subheader("📦 Competitor SKU Details")
        st.table(competitor_df.astype(str))

    # Number of matches toggle
    num_results = st.number_input("How many matching SKUs would you like to see?", min_value=1, max_value=100, value=5, step=1)

    result_df = find_matches(
        input_sku,
        brand_filter="ge" if search_type == "GE only" else "non-ge",
        top_n=num_results,
        strict=strict_config
    )

    if isinstance(result_df, pd.DataFrame):
        st.subheader("📊 Closest Matching Active SKUs")
        safe_dicts = [{k: str(v) for k, v in row.items()} for _, row in result_df.iterrows()]
        cleaned_df = pd.DataFrame(safe_dicts)
        st.table(cleaned_df)
    elif isinstance(result_df, str):
        st.warning(result_df)
    else:
        st.error("Unexpected result format.")
