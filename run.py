from src.data_processing import FeatureSelector
from src.app_dash import app, server

# #We need to explicitly specify the host in dashboard.py to access the dashboard app from outside the container
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)
