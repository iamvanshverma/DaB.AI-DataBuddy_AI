# DaB.AI - DataBuddy AI: The Smart Data Analysis Chatbot

**DaB.AI - DataBuddy AI** is a powerful, interactive web application built with **Python, Streamlit, and the Google Gemini API** that transforms raw CSV data into actionable business intelligence. It automates the entire Exploratory Data Analysis (EDA) pipeline—from data upload and quality checks to AI-powered insights and professional report generation—all driven by natural language queries.

---

## 🚀 Quick Glance

| Live Demo | Repository | Technology Stack | Export |
| :---: | :---: | :---: | :---: |
| [https://rhapxiueyuyygwmfwcjqns.streamlit.app/]() | [https://github.com/iamvanshverma/gfest.git]() | **Python** \| **Streamlit** \| **Gemini API** | **Sheets** / CSV |

---

## 🌟 Key Features

**DataBuddy AI** goes beyond a basic chatbot, offering a comprehensive, multi-layered suite of analytical tools:

### 🧠 AI-Powered Insights & Analysis

This is the core intelligence layer, powered by the **Gemini API**:

* **Natural Language Query Engine:** Ask complex data questions in plain English (e.g., "What are the main trends in my sales data?" or "Show me outliers in the revenue column").
* **Quick Analysis Presets:** Instantly generate a **Comprehensive Summary, Statistical Patterns, Best Visualizations,** and a **Data Quality Issues** report with a single click.
* **AI-Generated Statistical Insights:** Raw statistical tables are automatically interpreted and explained in clear, actionable language, making complex data easy to understand.
* **Suggested Follow-up Questions:** The AI provides continuous guidance by suggesting the next logical questions for deeper data exploration.

### 📊 Interactive Exploration & Visualizations

Tools for visual and manual data investigation:

* **Automated Overview Dashboard:** Instantly view high-level EDA plots upon data load, including **Distribution Plots** (Histograms, Box Plots) and a **Correlation Matrix**.
* **Custom Chart Builder:** Users can manually select granular **Chart Types** (e.g., Correlation Heatmap, Time Series, Multi-Variable Analysis), X-axis, Y-axis, and an optional Color column to create bespoke visualizations using libraries like **Plotly** and **Matplotlib**.
* **Categorical Data Analysis:** Dedicated tools for viewing **Bar Charts** and **Pie Charts** based on categorical columns like 'City' or 'Country'.

### 🔎 Advanced Data & Statistical Tools

The quantitative layer for rigorous data assessment:

* **Comprehensive Statistics:** Get detailed descriptive statistics including **Skewness, Kurtosis, Variance, IQR,** and **Coefficient of Variation (CV)** for all numeric columns, powered by **NumPy** for robust calculations.
* **Data Quality Assessment:** Dedicated analysis of **Missing Values**, including counts, percentages, and a **Missing Values Pattern** visualization plot.
* **Interactive Data Explorer:** Use **range sliders** for numeric columns and selection tools for categorical data to dynamically filter the dataset.
* **Analyze Filtered Data:** A powerful feature to run the AI-powered analysis *only* on the isolated data subset to investigate specific patterns or outliers.

### 📄 Reporting and Export Automation

The final layer for sharing and portability:

* **Comprehensive Report Generator:** Generate a **Full Data Report** that compiles all AI insights, statistics, and visualizations into a single, professional document (Implied PDF/HTML format).
* **Granular Data Export:** Download specific analysis components like **Summary Statistics, Correlation Data,** or the **Filtered Dataset** for external use.

---

## 🛠️ Technology Stack & Setup

DaB.AI is built on the following robust and scalable technologies:

| Category | Components | Function |
| :--- | :--- | :--- |
| **Frontend & Framework** | **Streamlit** | Provides the responsive, interactive, and beautifully designed user interface. |
| **Backend Intelligence** | **Google Gemini API** | Powers all complex reasoning, natural language understanding, and sophisticated data analysis. |
| **Language & Data Handling** | **Python, Pandas, NumPy** | Python is the core language; Pandas and NumPy handle efficient data loading, cleaning, manipulation, and mathematical operations. |
| **Visualization** | **Matplotlib, Plotly** | Core libraries used for generating static (Matplotlib) and interactive (Plotly) charts and dashboards. |

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository Link Here]
    cd dab.ai-databuddy-ai
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
