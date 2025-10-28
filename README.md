# 🚀 DaB.AI - DataBuddy AI: The Smart Data Analysis Chatbot

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Powered%20By-Gemini%20AI-blue?style=for-the-badge&logo=google)](https://ai.google.dev/models/gemini)
[![Streamlit](https://img.shields.io/badge/Deployed%20On-Streamlit-FF4B4B?style=for-for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Data Analysis](https://imgshields.io/badge/Data%20Analysis-EDA%20%26%20Stats-brightgreen?style=for-the-badge&logo=table)](https://pandas.pydata.org/)

**DaB.AI - DataBuddy AI** is a powerful, interactive web application built with **Python, Streamlit, and the Google Gemini API** that transforms raw CSV data into **actionable business intelligence**. It automates the entire Exploratory Data Analysis (EDA) pipeline—from data upload and quality checks to AI-powered insights and professional report generation—all driven by natural language queries.

---

## 🔗 Quick Access & Visual Preview

| Tool | Primary Use Case | Live Demo | Repository |
| :--- | :--- | :---: | :---: |
| **DaB.AI - DataBuddy AI** | Interactive EDA & Conversational Analysis (CSV Upload) | **[Live Application](https://rhapxiueyuyygwmfwcjqns.streamlit.app/)** | **[Source Code](https://github.com/iamvanshverma/gfest.git)** |

### 📸 Welcome Screen (Clickable Link)

Click the image below to launch the live application instantly:

| **DaB.AI - DataBuddy AI Homepage** |
| :---: |
| [![DataBuddy AI Welcome Screen](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Welcome%20Page.png)](https://rhapxiueyuyygwmfwcjqns.streamlit.app/) |

---

## ✨ DaB.AI - DataBuddy AI: Comprehensive Features

The platform provides a multi-layered suite of analytical tools, focusing purely on deep data exploration.

### I. 🧠 AI-Powered Insights & Analysis (The Chatbot)

The core intelligence layer, powered by the **Gemini API**, turning data tables into conversations.

| Feature | Detailed Description | Core Intelligence Layer |
| :--- | :--- | :--- |
| **Natural Language Engine** | Ask complex data questions in plain English (e.g., "What are the main trends in my sales data?"). | **Gemini API** |
| **Quick Analysis Presets** | Instantly generate a **Comprehensive Summary, Statistical Patterns,** or a **Data Quality Issues** report with a single click. | **Gemini API** |
| **Guided Exploration** | AI suggests **intelligent follow-up questions** based on its initial findings to encourage deeper investigation. | **Gemini API** |

| Chatbot Conversation Example (Input) | Chatbot Conversation Example (Output) |
| :---: | :---: |
| *Interface showing preset question in chat input:*<br>![Chatbot Interface](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Chatbot%20Image%201.png) | *AI-Generated Key Insights:*<br>![Chatbot Answer 2 - Insights](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Chatbot%20Answer%202%20(Insights).png) |

---

### II. 🔎 Advanced Data & Statistical Tools

The quantitative layer for rigorous data assessment and hands-on filtering.

| Feature | Detailed Description | Core Libraries |
| :--- | :--- | :--- |
| **Comprehensive Statistics** | Get detailed metrics including **Skewness, Kurtosis, Variance, IQR,** and **Coefficient of Variation (CV)** for all numeric columns. | **Pandas/NumPy** |
| **Data Quality Assessment** | Dedicated analysis of **Missing Values**, including counts, percentages, and a **Missing Values Pattern** visualization plot. | **Pandas** |
| **Interactive Data Explorer** | Use **range sliders** and selection tools to **dynamically filter** the dataset and run AI analysis *only* on the filtered subset. | **Pandas & Gemini API** |

| Statistical Analysis Example | Data Explorer Example |
| :---: | :---: |
| *Missing Values Pattern Plot:*<br>![Statistical Analysis 2 - Missing Values](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Statistical%20Analysis%202.png) | *The Data Explorer Tab:*<br>![Data Explorer tab](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Data%20Explorer%20tab.png) |

---

### III. 📊 Interactive Exploration & Reporting

The visualization and final export layer.

| Feature | Detailed Description | Visualization Tools |
| :--- | :--- | :--- |
| **Automated Dashboard** | Instantly view high-level EDA plots upon data load, including **Distribution Plots** and a **Correlation Matrix**. | **Plotly/Matplotlib** |
| **Custom Chart Builder** | Users can manually select granular **Chart Types** (e.g., Correlation Heatmap, Time Series), X-axis, Y-axis, and Color columns for bespoke visualizations. | **Plotly/Matplotlib** |
| **Comprehensive Report** | Generate a **Full Data Report** that compiles all AI insights, statistics, and visualizations into a single document. | **PDF/HTML (Implied)** |
| **Granular Export** | Download specific analysis components like **Summary Statistics, Correlation Data,** or the **Filtered Dataset**. | **CSV/DataFrames** |

| Custom Visualization Example | Summary Report Example |
| :---: | :---: |
| *Example of a Customized Visualization:*<br>![Customized Visualization](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Customized%20Visualization.png) | *Summary Report Tab - Visual Summary:*<br>![Summary Report 2 - Visual Summary](https://github.com/iamvanshverma/gfest/blob/5a5ac6c664d30abd10984f2beab3fa930f92d705/DaB.Ai%20DataBuddy%20ChatBot%20%26%20Analysis%20Images/Summary%20Report%202.png) |

---

## 🛠️ Technology Stack & Setup

The entire platform is built on the following robust and scalable technologies:

| Category | Components | Role |
| :--- | :--- | :--- |
| **Backend Intelligence** | **Google Gemini API** | Powers all complex reasoning, natural language understanding, and sophisticated data analysis. |
| **Frontend & Framework** | **Streamlit** | Provides the responsive, interactive, and beautifully designed user interface. |
| **Language & Data Handling** | **Python, Pandas, NumPy** | Core language; handles efficient data manipulation and mathematical operations. |
| **Visualization** | **Matplotlib, Plotly** | Core libraries used for generating static and interactive charts and dashboards. |

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/iamvanshverma/gfest.git](https://github.com/iamvanshverma/gfest.git)
    cd gfest
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up the Gemini API Key:**
    * Get your API key from [Google AI Studio]().
    * Set it as an environment variable (recommended):
        ```bash
        export GEMINI_API_KEY="YOUR_API_KEY_HERE"
        ```
    * *Alternatively, use Streamlit Secrets or a `.env` file.*
4.  **Run the application:**
    ```bash
    streamlit run app.py # (Or your primary Streamlit file name)
    ```

---

## 🚀 Getting Started (5-Step User Flow)

The application is designed for an intuitive, smooth user experience:

1.  **Upload:** Navigate to the **Settings & Upload** sidebar and upload your `.csv` file. *(Supported: CSV format, up to 200MB, UTF-8 Recommended)*.
2.  **Explore:** Immediately check the **Data Preview** and **Quick Stats** in the sidebar to understand data quality and basic dimensions.
3.  **Analyze (AI Chat Tab):** Use the **Quick Analysis** buttons (e.g., *Overall Summary*) or type your question, then click **Analyze**. Review the detailed Analysis Results and Key Insights provided by the AI.
4.  **Visualize (Visualizations Tab):** View the automated dashboard or use the **Interactive Visualizations** builder to create custom charts.
5.  **Report (Summary Report Tab):** Click **Generate Full Report** to automate the final output or use the **Export Options** to download specific analysis components.
