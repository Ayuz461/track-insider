📊 Insider Trading Tracker & Visualizer

A powerful and interactive Streamlit-based web app for tracking and analyzing insider trading activity using data from OpenInsider. The app features a clean UI, advanced visualizations, ML-based anomaly detection, risk scoring, and filtering to help identify and explore unusual or significant insider trades.

⸻

🚀 Features
	•	Dynamic Filtering: Filter trades by date range, trade type, ticker, insider name, or keyword search.
	•	Visual Dashboards:
	•	Summary statistics of insider trades.
	•	Price impact analysis around trade dates.
	•	Trade distribution and clustering charts.
	•	Anomaly Detection (ML): Uses Isolation Forest to detect trades that deviate from normal behavior based on price, volume, trade type, and recency.
	•	Risk Scoring System: Assigns a risk score to each trade based on volume, recency, and trade type.
	•	Export Options: Download filtered results as CSV or Excel files.
	•	Timeline Visualization: Track anomaly trends over time.
	•	Paginated Trade Table: View large filtered data efficiently.

⸻

⚙️ How to Run Locally

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/insider-trading-tracker.git
cd insider-trading-tracker

2. Install Dependencies

pip install -r requirements.txt

3. Launch the App

streamlit run app.py


⸻

📁 File Structure

insider-trading-tracker/
├── app.py                 # Main Streamlit application file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)


⸻

📦 Requirements
	•	Python 3.7+
	•	Streamlit
	•	pandas
	•	yfinance
	•	plotly
	•	scikit-learn
	•	xlsxwriter

Install all requirements easily:

pip install -r requirements.txt


⸻

📸 Screenshots

<img width="1277" alt="image" src="https://github.com/user-attachments/assets/2a4d049e-4a6f-4fd7-9c4b-461967826aa5" />
<img width="1279" alt="image" src="https://github.com/user-attachments/assets/80cd42df-ceb4-40fe-a077-b6ee2991d4f0" />



⸻

🙌 Credits
	•	Data Source: OpenInsider
	•	Developed using Streamlit, Plotly, scikit-learn, and yfinance

⸻

📌 Disclaimer

This tool is for educational and informational purposes only. It is not intended as financial advice or a trading recommendation.

⸻

🌐 License

MIT License
