# Deploying to Snowflake

## Step 1: Add Snowflake Connection

You need to configure your Snowflake connection first. Choose one method:

### Method 1: Interactive Connection Setup (Recommended)
```bash
snow connection add
```
This will prompt you for:
- Connection name (e.g., "default")
- Account name
- Username
- Password
- Warehouse
- Database
- Schema
- Role (optional)

### Method 2: Use Temporary Connection
You can deploy with temporary credentials using command-line parameters:
```bash
snow streamlit deploy --temporary-connection \
  --account YOUR_ACCOUNT \
  --user YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --warehouse COMPUTE_WH \
  --database YOUR_DATABASE \
  --schema YOUR_SCHEMA
```

## Step 2: Update snowflake.yml

Before deploying, update `snowflake.yml` with your actual Snowflake resources:

```yaml
definition_version: 1
native_app:
  name: my_streamlit_app
  version:
    name: v1
    patch: 0

streamlit:
  - name: STREAMLIT_APP
    file: streamlit_app.py
    stage: STREAMLIT_STAGE  # Update this to your stage name
    query_warehouse: COMPUTE_WH  # Update this to your warehouse
    environment:
      - name: ENV_VAR
        value: "value"
```

## Step 3: Deploy

Once connection is configured:
```bash
snow streamlit deploy
```

Or deploy with options:
```bash
snow streamlit deploy --replace --open
```

## Step 4: Get App URL

After deployment, get the URL:
```bash
snow streamlit get-url STREAMLIT_APP
```

## Troubleshooting

- **Connection issues**: Use `snow connection test` to verify your connection
- **Permission errors**: Ensure your user has CREATE STREAMLIT and CREATE STAGE privileges
- **Stage not found**: The deploy command will create the stage if it doesn't exist
