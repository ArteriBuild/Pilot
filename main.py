import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_policy_impact(policy_description, target_metric):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Analyze the potential impact of the following health policy on the specified wellbeing metric:
    Policy: {policy_description}
    Target Metric: {target_metric}
    Please provide:
    1. A brief summary of the potential impact
    2. Estimated magnitude of the impact (low, medium, high)
    3. Potential unintended consequences
    4. Suggestions for policy refinement
    Base your analysis on general knowledge of health policies and their typical impacts.
    """
    response = model.generate_content(prompt)
    return response.text

def policy_simulator():
    st.title("AI-Powered Policy Impact Simulator")
    st.write("This app simulates the potential impacts of health policies on various wellbeing metrics.")

    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []

    policy_description = st.text_area("Enter the health policy description:")
    target_metric = st.selectbox("Select the target wellbeing metric:", 
                                 ["Life expectancy", "Mental health", "Chronic conditions prevalence", 
                                  "Access to health services", "Healthcare costs"])

    if st.button("Analyze Impact"):
        if policy_description:
            with st.spinner("Analyzing policy impact..."):
                impact_analysis = analyze_policy_impact(policy_description, target_metric)
                st.subheader("Impact Analysis:")
                st.write(impact_analysis)

                # Save to history
                st.session_state.history.append({
                    "policy": policy_description,
                    "metric": target_metric,
                    "analysis": impact_analysis
                })
        else:
            st.warning("Please enter a policy description.")

    # Display history
    if st.session_state.history:
        st.subheader("Analysis History")
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analysis {len(st.session_state.history) - i}: {item['metric']}"):
                st.write(f"Policy: {item['policy']}")
                st.write(f"Analysis: {item['analysis']}")

def fetch_abs_data(dataset_id, query_params):
    base_url = f"https://api.data.abs.gov.au/data/{dataset_id}"
    try:
        response = requests.get(base_url, params=query_params, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.json()
    except requests.RequestException as e:
        st.warning(f"Failed to fetch data for dataset: {dataset_id}. Using mock data instead. Error: {str(e)}")
        return None

def generate_mock_life_expectancy_data():
    years = range(2010, 2022)
    male_data = [80 + np.random.normal(0, 0.5) for _ in years]
    female_data = [84 + np.random.normal(0, 0.5) for _ in years]

    data = []
    for year, male, female in zip(years, male_data, female_data):
        data.append({'Year': year, 'Sex': 'Male', 'Value': male})
        data.append({'Year': year, 'Sex': 'Female', 'Value': female})

    return pd.DataFrame(data)

def generate_mock_population_data():
    dates = pd.date_range(start='2010-01-01', end='2022-12-31', freq='Q')
    population = [23000000 + i * 100000 + np.random.normal(0, 50000) for i in range(len(dates))]

    return pd.DataFrame({'Date': dates, 'Value': population})

def generate_data_analysis(life_exp_df, population_df):
    model = genai.GenerativeModel('gemini-pro')

    # Prepare summary statistics
    life_exp_summary = life_exp_df.groupby('Sex')['Value'].agg(['mean', 'min', 'max']).to_string()
    population_summary = population_df['Value'].agg(['mean', 'min', 'max']).to_string()

    prompt = f"""
    Analyze the following data on life expectancy and population in Australia:

    Life Expectancy Summary:
    {life_exp_summary}

    Population Summary:
    {population_summary}

    Please provide:
    1. A brief overview of the trends in life expectancy and population.
    2. Any notable observations or patterns in the data.
    3. Potential implications for public health and social policies.
    4. Suggestions for areas that might need further investigation or attention.

    Limit your response to about 250 words.
    """

    response = model.generate_content(prompt)
    return response.text

def wellbeing_dashboard():
    st.title("Australian Wellbeing Dashboard")

    # Fetch life expectancy data
    life_exp_data = fetch_abs_data(
        "LXYR",
        {
            "format": "jsondata",
            "LXMEASURE": "1",  # Life expectancy at birth
            "SEX": "1+3",  # Males and Females
            "REGIONTYPE": "AUS",  # Australia-wide
            "FREQUENCY": "A",  # Annual
            "MEASURE": "2",  # Years
            "start-period": "2010",
            "end-period": "2021"
        }
    )

    if life_exp_data:
        life_exp_df = pd.DataFrame(life_exp_data['dataSets'][0]['observations']).T
        life_exp_df.columns = ['Value']
        life_exp_df['Year'] = [item[4] for item in life_exp_data['structure']['dimensions']['observation'][4]['values']]
        life_exp_df['Sex'] = [item[1] for item in life_exp_data['structure']['dimensions']['observation'][1]['values']]
        life_exp_df['Value'] = life_exp_df['Value'].astype(float)
    else:
        life_exp_df = generate_mock_life_expectancy_data()

    # Fetch population data
    pop_data = fetch_abs_data(
        "ERP_QUARTERLY",
        {
            "format": "jsondata",
            "REGIONTYPE": "AUS",  # Australia-wide
            "FREQUENCY": "Q",  # Quarterly
            "MEASURE": "1",  # Number
            "SEX": "3",  # Persons
            "AGE": "999",  # All ages
            "start-period": "2010",
            "end-period": "2022"
        }
    )

    if pop_data:
        population_df = pd.DataFrame(pop_data['dataSets'][0]['observations']).T
        population_df.columns = ['Value']
        population_df['Date'] = [item[5] for item in pop_data['structure']['dimensions']['observation'][5]['values']]
        population_df['Value'] = population_df['Value'].astype(float)
        population_df['Date'] = pd.to_datetime(population_df['Date'])
    else:
        population_df = generate_mock_population_data()

    # Generate and display AI analysis
    with st.spinner("Generating data analysis..."):
        analysis = generate_data_analysis(life_exp_df, population_df)
        st.subheader("AI-Generated Data Analysis")
        st.write(analysis)

    # Display charts
    st.subheader("Life Expectancy in Australia")
    fig = px.line(life_exp_df, x="Year", y="Value", color="Sex", 
                  title="Life Expectancy at Birth in Australia")
    fig.update_layout(yaxis_title="Life Expectancy (Years)")
    st.plotly_chart(fig)

    st.subheader("Population Growth in Australia")
    fig = px.line(population_df, x="Date", y="Value", 
                  title="Total Population of Australia Over Time")
    fig.update_layout(yaxis_title="Population")
    st.plotly_chart(fig)

def main():
    # Add a placeholder logo
    try:
        logo = Image.open("logo.png")  # Replace with your actual logo file
        st.image(logo, width=200)
    except FileNotFoundError:
        st.write("Placeholder for logo")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Policy Simulator", "Wellbeing Dashboard"])

    if page == "Policy Simulator":
        policy_simulator()
    elif page == "Wellbeing Dashboard":
        wellbeing_dashboard()

if __name__ == "__main__":
    main()
