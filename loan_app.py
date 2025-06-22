import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st
from PIL import Image
#from streamlit_chat import message
# Set page config
#st.set_page_config(page_title="Loan App", layout="wide", initial_sidebar_state="expanded")

# Apply custom frame styling ONLY on apply page
# if 'page' in st.session_state and st.session_state.page == 'apply':
#     st.markdown("""
#         <style>
#         .block-container {
#             background-color: white;
#             border-radius: 15px;
#             padding: 2rem;
#             box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
#             margin: 30px;
#         }
#         </style>
#     """, unsafe_allow_html=True)
#import openai
# Apply custom app styling
# Load and preprocess data
def load_data():
    df = pd.read_csv('data/loan_data_set.csv')
    df['Income_Group'] = pd.cut(df['ApplicantIncome'],
                                bins=[0, 2500, 5000, 7500, 10000, 20000, df['ApplicantIncome'].max()],
                                labels=['<2.5k', '2.5k–5k', '5k–7.5k', '7.5k–10k', '10k–20k', '20k+'])
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert Loan_Status to binary
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Clean Dependents column
    df['Dependents'] = df['Dependents'].replace('3+', '3')
    df['Dependents'] = df['Dependents'].fillna('0')
    df['Dependents'] = df['Dependents'].astype(int)
    
    # Clean other categorical columns
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    for col in cat_cols:
        df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
    
    return df

# Feature engineering
def feature_engineering(df):
    # Total income
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # EMI (assuming 8.5% interest rate)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['EMI'] = df['LoanAmount'] * 0.0085 * (1 + 0.0085)**df['Loan_Amount_Term'] / ((1 + 0.0085)**df['Loan_Amount_Term'] - 1)
    
    # Income to EMI ratio
    df['Income_to_EMI'] = df['TotalIncome'] / (df['EMI'] + 1e-6)
    
    # Loan amount to income ratio
    df['Loan_to_Income'] = df['LoanAmount'] / (df['TotalIncome'] + 1e-6)
    
    return df

# Train model
def train_model(df):
    # Features and target
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    # Define columns
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                'TotalIncome', 'EMI', 'Income_to_EMI', 'Loan_to_Income']
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Credit_History', 'Property_Area']
    
    # Preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])
    
    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    return model

# Save and load model
def save_model(model, filename='loan_model.pkl'):
    joblib.dump(model, filename)

def load_model(filename='loan_model.pkl'):
    return joblib.load(filename)

# Streamlit app
def run_streamlit_app(model):
    st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.stop()
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background-color: #ffffff;
    }
    .stNumberInput>div>div>input {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏦 AI-Powered Loan Approval Prediction System")
    st.markdown("""
    This system predicts whether a loan application will be approved based on applicant information. 
    Fill in the details below to get a prediction and personalized advice.
    """)
    
    # Initialize chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info("""
    This application uses machine learning to predict loan approvals. 
    The model was trained on historical loan application data.
    """)
    
    # Main form
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", [0, 1, 2, 3])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            
        with col2:
            st.subheader("Financial Information")
            applicant_income = st.number_input("Applicant Income (Monthly)", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income (Monthly)", min_value=0)
            loan_amount = st.number_input("Loan Amount (Thousands)", min_value=0)
            loan_term = st.selectbox("Loan Term (Months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
            credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        submitted = st.form_submit_button("Predict Loan Approval")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })
        
        # Add engineered features
        input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
        input_data['EMI'] = input_data['LoanAmount'] * 0.0085 * (1 + 0.0085)**input_data['Loan_Amount_Term'] / ((1 + 0.0085)**input_data['Loan_Amount_Term'] - 1)
        input_data['Income_to_EMI'] = input_data['TotalIncome'] / (input_data['EMI'] + 1e-6)
        input_data['Loan_to_Income'] = input_data['LoanAmount'] / (input_data['TotalIncome'] + 1e-6)
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.success(f"✅ Loan Approved (Probability: {probability[1]*100:.1f}%)")
            
            # Generate AI advice for approved loans
            advice_prompt = f"""
            The loan application has been approved with a {probability[1]*100:.1f}% probability. 
            The applicant is a {gender}, {married}, with {dependents} dependents and {education} education.
            Their total monthly income is {input_data['TotalIncome'].values[0]:.2f} and they requested a loan of {loan_amount} for {loan_term} months.
            Provide 3-4 bullet points of constructive advice for the applicant regarding managing their loan responsibly.
            """
            
            ai_advice = get_ai_response(advice_prompt)
            st.markdown("### 💡 AI-Powered Advice")
            st.markdown(ai_advice)
        else:
            st.error(f"❌ Loan Not Approved (Probability: {probability[0]*100:.1f}%)")
            
            # Generate AI advice for rejected loans
            advice_prompt = f"""
            The loan application was not approved (rejection probability: {probability[0]*100:.1f}%). 
            The applicant is a {gender}, {married}, with {dependents} dependents and {education} education.
            Their total monthly income is {input_data['TotalIncome'].values[0]:.2f} and they requested a loan of {loan_amount} for {loan_term} months.
            Provide 3-4 bullet points of constructive advice for the applicant on how they might improve their chances of approval in the future.
            """
            
            ai_advice = get_ai_response(advice_prompt)
            st.markdown("### 💡 AI-Powered Advice")
            st.markdown(ai_advice)
        
        # Show key metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Income to EMI Ratio", f"{input_data['Income_to_EMI'].values[0]:.1f}", 
                      help="Higher is better (recommended > 3)")
        
        with col2:
            st.metric("Loan to Income Ratio", f"{input_data['Loan_to_Income'].values[0]:.1f}", 
                      help="Lower is better (recommended < 5)")
        
        with col3:
            st.metric("Total Income", f"${input_data['TotalIncome'].values[0]:,.2f}")
    
    # AI Chatbot section
    st.markdown("---")
    st.subheader("🤖 Loan Advisor Chat")
    st.markdown("""
    <style>
    .stTextInput>div>div>input {
        color: black !important;
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
    user_input = st.text_input("Ask any question about loans, eligibility, or financial advice:", key="input")
    
    if user_input:
        response = get_ai_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    #if st.session_state['generated']:
   #    for i in range(len(st.session_state['generated'])-1, -1, -1):
    #        message(st.session_state["generated"][i], key=str(i))
    #        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    # if st.session_state['generated']:
    #     for i in range(len(st.session_state['generated'])-1, -1, -1):
    #         st.markdown(f'<div style="color: black; font-weight: bold; margin-bottom: 5px;">🧑‍💼 You: {st.session_state["past"][i]}</div>', unsafe_allow_html=True)
    #         st.markdown(f'<div style="color: #444; background-color: #f1f1f1; padding: 8px; border-radius: 5px; margin-bottom: 20px;">🤖 {st.session_state["generated"][i]}</div>', unsafe_allow_html=True)
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.markdown(f"""
            <div style='text-align: right; background-color: #f1f1f1; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;'>
                <b>You:</b> {st.session_state['past'][i]}
            </div>
            <div style='text-align: left; background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin: 5px 0;color: black;'>
                <b>Advisor:</b> {st.session_state["generated"][i]}
            </div>
            """, unsafe_allow_html=True)
# AI response function (using OpenAI)
def get_ai_response(prompt):
    try:
        # In a real app, you would use your OpenAI API key
        # openai.api_key = st.secrets["openai_key"]
        
        # For demo purposes, we'll use a simplified response
        simplified_responses = {
            "approved": """
            - Make sure to budget carefully to accommodate your EMI payments each month
            - Consider setting up automatic payments to avoid missing due dates
            - Maintain or improve your credit score by paying all bills on time
            - Keep your debt-to-income ratio low by avoiding additional large loans
            """,
            "rejected": """
            - Work on improving your credit score by paying existing debts on time
            - Consider reducing your loan amount request or applying with a longer term
            - Try to increase your income or add a co-applicant with stable income
            - Wait for 6 months and reapply after improving your financial situation
            """
        }
        
        if "approved" in prompt.lower():
            return simplified_responses["approved"]
        elif "not approved" in prompt.lower() or "rejection" in prompt.lower():
            return simplified_responses["rejected"]
        else:
            return "I'm a demo AI and can provide general advice. For approved loans: focus on timely repayments. For rejections: work on improving credit score and reducing debt."
            
        # Real implementation would use:
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        
    except Exception as e:
        return f"AI service is currently unavailable. Error: {str(e)}"

# # Interactive front page
image = Image.open("Ai-image.jpg")
def show_front_page():
    st.set_page_config(page_title="Loan App Home", layout="centered")

    st.markdown("""
        <style>
        .big-font {
            font-size:50px !important;
            font-weight: bold;
            text-align: center;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stButton>button {
            width: 100%;
            font-size: 20px;
            padding: 0.5em;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">🏦 Welcome to AI-Powered Loan Predictor</p>', unsafe_allow_html=True)
    st.write("      This application uses machine learning to predict loan approval chances based on applicant details.")
    col1, col2, col3 = st.columns([1, 2, 1])  # 3 columns, center one is wider
    with col2:
        st.image(image, width=400, caption="AI-Powered Loan Prediction")
    st.markdown("---")
    if st.button("📊 Go to Dashboard"):
        st.session_state.page = 'dashboard'
    if st.button("➡️ Go to Prediction System"):
        st.session_state.page = "predict"
    # # ✅ Pie Chart: Loan Status Overview
    # st.markdown("### 🥧 Loan Approval Status Overview")
    # status_counts = df['Loan_Status'].value_counts()
    # labels = status_counts.index
    # sizes = status_counts.values
    # colors = ['#4CAF50', '#F44336']

    # fig, ax = plt.subplots()
    # ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    # ax.axis('equal')

    # # Optional: Center it like the image
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     st.pyplot(fig)
def show_dashboard(df, model):
    st.markdown('<h1 style="text-align:center;">📊 Loan Insights Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("-----")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🥧 Loan Approval Distribution")
        status_counts = df['Loan_Status'].replace({1: 'Approved', 0: 'Not Approved'}).value_counts()
        # Define custom colors
        colors = ['#20B2AA', '#FF7F50']  # dark green for approved, tomato red for not approved

    # Create pie chart
        fig1, ax1 = plt.subplots(figsize=(6, 7))
        ax1.pie(
        status_counts,
        labels=status_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12, 'color': 'black'}
        )
        ax1.axis('equal')  # Equal aspect ratio ensures the pie is circular

        st.pyplot(fig1)
        #status_counts = df['Loan_Status'].value_counts()
        # fig1, ax1 = plt.subplots(figsize=(5, 6))
        # ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
        # ax1.axis('equal')
        # st.pyplot(fig1)

    with col2:
        st.markdown("#### 📈 Avg Loan Amount by Income Range")
        group_data = df.groupby('Income_Group')['LoanAmount'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(group_data['Income_Group'], group_data['LoanAmount'], marker='o', linestyle='-', color='blue')
        ax2.set_xlabel("Applicant Income Range")
        ax2.set_ylabel("Average Loan Amount")
        ax2.set_title("Loan Amount by Income Group")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

# Charts Row 2: Property Area + Education
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🏩 Approval by Property Area")
        area_data = df.groupby('Property_Area')['Loan_Status'].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=area_data, x='Property_Area', y='Loan_Status', palette='pastel', ax=ax3)
        ax3.set_ylabel("Approval Rate")
        st.pyplot(fig3)
    with col4:
        st.markdown("#### 🎓 Education vs Loan Status")
        edu_status = pd.crosstab(df['Education'], df['Loan_Status'], normalize='index') * 100
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        edu_status.plot(kind='bar', stacked=True, color=['#F44336', '#4CAF50'], ax=ax4)
        ax4.set_ylabel("Percentage")
        ax4.set_title("Loan Status by Education")
        st.pyplot(fig4)

# Charts Row 3: Income Box + Loan Dist + Credit
    col5, col6,= st.columns(2)
    with col5:
        st.markdown("#### 💳 Credit History Impact")
        credit_approval = df.groupby('Credit_History')['Loan_Status'].mean().reset_index()
        fig7, ax7 = plt.subplots(figsize=(6, 5))
        sns.barplot(data=credit_approval, x='Credit_History', y='Loan_Status', palette='Blues', ax=ax7)
        ax7.set_title("Approval Rate by Credit History")
        ax7.set_ylabel("Approval Probability")
        st.pyplot(fig7)

    with col6:
        st.markdown("#### 💰 Loan Amount Distribution")
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        sns.histplot(df['LoanAmount'], bins=30, kde=True, color='purple', ax=ax6)
        ax6.set_title("Loan Amount Distribution")
        st.pyplot(fig6)

    st.markdown("---")
    if st.button("🏠 Back to Home"):
        st.session_state.page = 'home'

# # Main launcher
# if __name__ == "__main__":
#     print("Loading and preprocessing data...")
#     df = load_data()
#     df = feature_engineering(df)

#     print("Training model...")
#     model = train_model(df)
#     save_model(model)

#     print("Loading saved model...")
#     model = load_model()

#     # Page handling
#     if 'page' not in st.session_state:
#         st.session_state.page = 'home'

#     if st.session_state.page == 'home':
#         show_front_page()
#     elif st.session_state.page == 'predict':
#         run_streamlit_app(model)

# Main launcher
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    df = feature_engineering(df)

    print("Training model...")
    model = train_model(df)
    save_model(model)

    print("Loading saved model...")
    model = load_model()

    # Page handling
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        show_front_page()
    elif st.session_state.page == 'dashboard':
        show_dashboard(df, model)
    elif st.session_state.page == 'predict':
        run_streamlit_app(model)
    
