# LangGraph-Multi-Agent-Chatbot


### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Required Files
Ensure these files are in your project directory:
- `model.py` (your existing model file)
- `metadata_adv.txt` (database schema)
- `query_example.txt` (SQL examples)
- `Data/` folder with CSV files:
  - `customers.csv`
  - `products.csv`
  - `sales.csv`
  - `employees.csv`
  - `productsubcategories.csv`
  - `productcategories.csv`
  - `vendors.csv`
  - `vendorproduct.csv`

### 4. Run the Dashboard
```bash
streamlit run updated_langgraph_integration.py
```

## 🎯 Features

- **Multi-Agent Orchestration**: Uses LangGraph for sophisticated agent coordination
- **Persona-Aware Analysis**: Adapts insights and KPIs based on user role
- **Adventure Works Integration**: Works with your existing database and model
- **Real-time SQL Generation**: Creates optimized queries for each persona
- **Interactive Dashboard**: Streamlit-based UI with visualization support

## 🛠️ Architecture

The system uses a multi-agent workflow:
1. **Persona Router** → Determines user role
2. **SQL Generator** → Creates persona-enhanced queries
3. **SQL Executor** → Runs queries on your data
4. **Insights Generator** → Creates role-specific analysis
5. **Response Formatter** → Formats output for the persona

