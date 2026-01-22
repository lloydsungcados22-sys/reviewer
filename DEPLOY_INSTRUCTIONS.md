# Quick Deployment Guide

## Option 1: Add Connection Then Deploy (Recommended)

### Step 1: Add Snowflake Connection
Run this command and enter your credentials when prompted:
```bash
cd "C:\Users\almar\OneDrive\Documents\Streamlit DF\my_streamlit_app"
snow connection add --connection-name default --default
```

You'll be prompted for:
- **Account**: Your Snowflake account identifier (e.g., `xy12345`)
- **Username**: Your Snowflake username
- **Password**: Your Snowflake password
- **Warehouse**: Your warehouse name (e.g., `COMPUTE_WH`)
- **Database**: Your database name
- **Schema**: Your schema name
- **Role**: Your role (optional)

### Step 2: Update snowflake.yml
Edit `snowflake.yml` and update:
- `stage: STREAMLIT_STAGE` → Your stage name (or leave as is, it will be created)
- `query_warehouse: COMPUTE_WH` → Your actual warehouse name

### Step 3: Deploy
```bash
snow streamlit deploy
```

Or with options:
```bash
snow streamlit deploy --replace --open
```

## Option 2: Deploy with Temporary Connection (One-time)

If you don't want to save credentials, use temporary connection:

```bash
cd "C:\Users\almar\OneDrive\Documents\Streamlit DF\my_streamlit_app"
snow streamlit deploy --temporary-connection \
  --account YOUR_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse YOUR_WAREHOUSE \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA
```

Replace the placeholders with your actual Snowflake credentials.

## After Deployment

Get your app URL:
```bash
snow streamlit get-url STREAMLIT_APP
```

List all Streamlit apps:
```bash
snow streamlit list
```

## Troubleshooting

- **Test connection**: `snow connection test`
- **List connections**: `snow connection list`
- **Check deployment status**: `snow streamlit describe STREAMLIT_APP`
