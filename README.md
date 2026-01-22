# My Streamlit App (Snowflake)

A Streamlit application configured for Snowflake deployment.

## Project Structure

- `streamlit_app.py` - Main Streamlit application file
- `snowflake.yml` - Snowflake project configuration
- `environment.yml` - Python environment dependencies
- `requirements.txt` - Additional Python packages

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Snowflake connection:**
   - Create `.streamlit/secrets.toml` file
   - Add your Snowflake credentials:
   ```toml
   [snowflake]
   user = "your_username"
   password = "your_password"
   account = "your_account"
   warehouse = "COMPUTE_WH"
   database = "your_database"
   schema = "your_schema"
   ```

## Local Development

Run the app locally:
```bash
streamlit run streamlit_app.py
```

## Deploy to Snowflake

1. **Login to Snowflake:**
   ```bash
   snow login
   ```

2. **Deploy the app:**
   ```bash
   snow streamlit deploy
   ```

## Configuration

Edit `snowflake.yml` to configure:
- App name and version
- Query warehouse
- Stage location
- Environment variables
