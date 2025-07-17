"""
UPDATED LangGraph Multi-Agent Integration
This file integrates the latest LangGraph API with model.py components.
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
from enum import Enum
import operator

# Current LangGraph imports (2024)
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

# Import your existing model.py components
try:
    from model import (
        sql_llm_model,
        run_inference, 
        run_summary_inference,
        sql_query_execution,
        func_final_result,
        # Import your data
        customers, products, productsubcategory, 
        productcategory, vendor, productvendor, 
        employee, sales
    )
    st.success("✅ Successfully imported your existing model.py components!")
except ImportError as e:
    st.error(f"❌ Could not import from model.py: {e}")
    # For demo purposes, create fallback functions
    def func_final_result(query):
        return ["Demo response", "SELECT 'demo' as result"]
    def sql_query_execution(query):
        return pd.DataFrame({'result': ['Demo data']})

# =============================================================================
# STATE DEFINITION FOR LANGGRAPH
# =============================================================================

class AgentState(TypedDict):
    """State object for the multi-agent system using current LangGraph API"""
    # Core workflow data
    user_query: str
    selected_persona: Optional[str]
    detected_persona: str
    confidence_score: float
    
    # Processing results
    sql_query: str
    data_results: Optional[pd.DataFrame]
    
    # Agent outputs
    insights: List[str]
    recommendations: List[str]
    kpis: Dict[str, Any]
    
    # Workflow control
    next_action: str
    error_message: Optional[str]
    
    # Chat messages (with reducer to append messages)
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]

# =============================================================================
# PERSONA TYPES AND CONFIGURATIONS
# =============================================================================

class PersonaType(Enum):
    SALES_MANAGER = "sales_manager"
    PRODUCT_ANALYST = "product_analyst"
    CUSTOMER_SUCCESS = "customer_success"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

# Persona configurations based on your Adventure Works data
PERSONA_CONFIGS = {
    PersonaType.SALES_MANAGER: {
        "name": "Sales Manager",
        "icon": "📈",
        "focus": "Revenue, territories, sales performance, team metrics",
        "prompt_enhancement": """
        Focus on sales performance metrics including:
        - Revenue by territory (use employee.Territory and sales.LineTotal)
        - Sales rep performance (use employee.FullName and sum of sales)
        - Monthly/quarterly trends (use sales.OrderDate)
        - Customer acquisition metrics
        - Average deal sizes and close rates
        """,
        "kpis": ["Total Revenue", "Territory Performance", "Sales Growth", "Team Performance"]
    },
    
    PersonaType.PRODUCT_ANALYST: {
        "name": "Product Analyst",
        "icon": "📊", 
        "focus": "Product margins, inventory, category performance",
        "prompt_enhancement": """
        Focus on product analytics including:
        - Product profitability (use products.StandardCost vs sales.LineTotal)
        - Category performance (use productcategory and productsubcategory)
        - Inventory analysis (use products.ListPrice and sales.OrderQty)
        - Margin calculations (LineTotal - (OrderQty * StandardCost))
        - Top/bottom performing products
        """,
        "kpis": ["Product Profitability", "Margin Analysis", "Category Performance", "Inventory Metrics"]
    },
    
    PersonaType.CUSTOMER_SUCCESS: {
        "name": "Customer Success Lead",
        "icon": "👥",
        "focus": "Customer relationships, retention, lifetime value",
        "prompt_enhancement": """
        Focus on customer relationship metrics including:
        - Customer lifetime value (use customers and sum of sales.LineTotal)
        - Purchase frequency and patterns (use sales.OrderDate)
        - Customer segmentation by value
        - Repeat purchase analysis
        - Customer tenure and retention
        """,
        "kpis": ["Customer Lifetime Value", "Retention Rate", "Customer Satisfaction", "Repeat Purchases"]
    },
    
    PersonaType.EXECUTIVE: {
        "name": "Executive",
        "icon": "🎯",
        "focus": "Strategic overview, growth trends, high-level KPIs",
        "prompt_enhancement": """
        Focus on strategic overview metrics including:
        - Overall business performance (total revenue, growth)
        - High-level KPIs across all departments
        - Year-over-year comparisons
        - Strategic insights and trends
        - Executive dashboard metrics
        """,
        "kpis": ["Total Revenue", "Growth Rate", "Market Performance", "Strategic KPIs"]
    }
}

# =============================================================================
# AGENT NODE FUNCTIONS
# =============================================================================

def persona_router_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Route based on persona selection or detection"""
    
    # If persona is manually selected, use it
    if state.get("selected_persona"):
        return {
            "detected_persona": state["selected_persona"],
            "confidence_score": 1.0,
            "next_action": "generate_sql"
        }
    else:
        # Perform automatic detection using your existing LLM
        try:
            detection_prompt = f"""
            Analyze this business query and determine the most likely user persona:
            
            Query: {state['user_query']}
            
            Personas:
            1. sales_manager - Revenue, territories, sales performance
            2. product_analyst - Product margins, inventory, category performance  
            3. customer_success - Customer retention, lifetime value, satisfaction
            4. executive - Strategic overview, growth trends, high-level metrics
            
            Respond with just the persona name (sales_manager, product_analyst, customer_success, or executive).
            """
            
            # Use your existing LLM model
            response = sql_llm_model.invoke([HumanMessage(content=detection_prompt)])
            detected_persona = response.content.strip().lower()
            
            # Validate the detected persona
            valid_personas = [p.value for p in PersonaType if p != PersonaType.UNKNOWN]
            if detected_persona not in valid_personas:
                detected_persona = "unknown"
                confidence = 0.0
            else:
                confidence = 0.8  # High confidence for valid detection
            
            return {
                "detected_persona": detected_persona,
                "confidence_score": confidence,
                "next_action": "generate_sql" if confidence > 0.6 else "request_clarification"
            }
            
        except Exception as e:
            return {
                "detected_persona": "unknown",
                "confidence_score": 0.0,
                "next_action": "handle_error",
                "error_message": f"Persona detection failed: {str(e)}"
            }

def sql_generator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate SQL using your existing model.py functions with persona enhancement"""
    
    try:
        # Get persona configuration
        persona_type = PersonaType(state["detected_persona"]) if state["detected_persona"] != "unknown" else PersonaType.UNKNOWN
        
        # Enhance query with persona context
        if persona_type != PersonaType.UNKNOWN:
            config = PERSONA_CONFIGS[persona_type]
            enhanced_query = f"""
            User Query: {state['user_query']}
            
            Persona Context: You are answering for a {config['name']} who needs insights focused on:
            {config['prompt_enhancement']}
            
            When generating SQL, prioritize metrics and analyses that are most relevant to a {config['name']}.
            """
        else:
            enhanced_query = state['user_query']
        
        # Use your existing func_final_result function
        result = func_final_result(enhanced_query)
        
        # Extract SQL query (result[1] contains the SQL)
        sql_query = result[1] if len(result) > 1 else "SELECT 'No SQL generated' as message"
        
        return {
            "sql_query": sql_query,
            "next_action": "execute_sql"
        }
        
    except Exception as e:
        return {
            "sql_query": "",
            "next_action": "handle_error",
            "error_message": f"SQL generation failed: {str(e)}"
        }

def sql_executor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute SQL using your existing sql_query_execution function"""
    
    try:
        # Use your existing SQL execution function
        data_results = sql_query_execution(state["sql_query"])
        
        if data_results.empty:
            return {
                "data_results": data_results,
                "next_action": "generate_empty_response"
            }
        else:
            return {
                "data_results": data_results,
                "next_action": "generate_insights"
            }
            
    except Exception as e:
        return {
            "data_results": pd.DataFrame(),
            "next_action": "handle_error",
            "error_message": f"SQL execution failed: {str(e)}"
        }

def insights_generator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate persona-specific insights using your existing functions"""
    
    try:
        persona_type = PersonaType(state["detected_persona"]) if state["detected_persona"] != "unknown" else PersonaType.UNKNOWN
        data = state["data_results"]
        
        # Use your existing run_summary_inference function
        if persona_type != PersonaType.UNKNOWN:
            persona_config = PERSONA_CONFIGS[persona_type]
            persona_context = f"""
            Generate insights for a {persona_config['name']} based on this data.
            Focus on: {persona_config['focus']}
            Key metrics to highlight: {', '.join(persona_config['kpis'])}
            """
            summary = run_summary_inference(data, state["sql_query"], state["user_query"] + persona_context)
        else:
            summary = run_summary_inference(data, state["sql_query"], state["user_query"])
        
        # Generate persona-specific KPIs
        kpis = generate_persona_kpis(data, persona_type)
        
        # Generate recommendations
        recommendations = generate_persona_recommendations(data, persona_type)
        
        return {
            "insights": [summary],
            "kpis": kpis,
            "recommendations": recommendations,
            "next_action": "format_response"
        }
        
    except Exception as e:
        return {
            "insights": [f"Error generating insights: {str(e)}"],
            "kpis": {},
            "recommendations": [],
            "next_action": "handle_error",
            "error_message": f"Insights generation failed: {str(e)}"
        }

def response_formatter_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Format final response for the persona"""
    
    persona_type = PersonaType(state["detected_persona"]) if state["detected_persona"] != "unknown" else PersonaType.UNKNOWN
    
    if persona_type != PersonaType.UNKNOWN:
        persona_config = PERSONA_CONFIGS[persona_type]
        formatted_response = f"""
        ## {persona_config['icon']} {persona_config['name']} Analysis
        
        **Query:** {state['user_query']}
        
        **Key Insights:**
        {chr(10).join(['• ' + insight for insight in state['insights']])}
        
        **Recommendations:**
        {chr(10).join(['• ' + rec for rec in state['recommendations']])}
        
        **Focus Areas:** {persona_config['focus']}
        """
    else:
        formatted_response = f"""
        ## 📋 Business Analysis
        
        **Query:** {state['user_query']}
        
        **Analysis Results:**
        {chr(10).join(state['insights'])}
        """
    
    return {
        "messages": [AIMessage(content=formatted_response)],
        "next_action": "complete"
    }

def error_handler_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Handle errors in the workflow"""
    error_msg = state.get("error_message", "Unknown error occurred")
    
    return {
        "messages": [AIMessage(content=f"❌ Error: {error_msg}")],
        "next_action": "complete"
    }

def clarification_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Request clarification when persona detection confidence is low"""
    clarification_msg = """
    I'm not sure which role you're asking from. Could you clarify:
    - Are you asking about sales performance? 📈
    - Product analysis and margins? 📊  
    - Customer success metrics? 👥
    - Executive/strategic overview? 🎯
    
    Or please select your role from the dropdown above.
    """
    
    return {
        "messages": [AIMessage(content=clarification_msg)],
        "next_action": "complete"
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_persona_kpis(data: pd.DataFrame, persona_type: PersonaType) -> Dict[str, Any]:
    """Generate persona-specific KPIs"""
    
    if data.empty:
        return {}
    
    kpis = {}
    
    try:
        if persona_type == PersonaType.SALES_MANAGER:
            # Sales KPIs
            if 'total_revenue' in data.columns:
                kpis['Total Revenue'] = f"${data['total_revenue'].sum():,.2f}"
            if 'Territory' in data.columns:
                kpis['Territories'] = data['Territory'].nunique()
            if 'avg_deal_size' in data.columns:
                kpis['Avg Deal Size'] = f"${data['avg_deal_size'].mean():,.2f}"
        
        elif persona_type == PersonaType.PRODUCT_ANALYST:
            # Product KPIs
            if 'ProductName' in data.columns:
                kpis['Products Analyzed'] = data['ProductName'].nunique()
            if any('profit' in col.lower() for col in data.columns):
                profit_col = next((col for col in data.columns if 'profit' in col.lower()), None)
                if profit_col:
                    kpis['Total Profit'] = f"${data[profit_col].sum():,.2f}"
        
        elif persona_type == PersonaType.CUSTOMER_SUCCESS:
            # Customer KPIs
            customer_cols = ['CustomerID', 'FullName']
            customer_col = next((col for col in customer_cols if col in data.columns), None)
            if customer_col:
                kpis['Customers'] = data[customer_col].nunique()
            if any('value' in col.lower() for col in data.columns):
                value_col = next((col for col in data.columns if 'value' in col.lower()), None)
                if value_col:
                    kpis['Avg Customer Value'] = f"${data[value_col].mean():,.2f}"
        
        elif persona_type == PersonaType.EXECUTIVE:
            # Executive KPIs
            numeric_cols = data.select_dtypes(include=['number']).columns
            for col in numeric_cols[:3]:  # Top 3 metrics
                kpis[col.replace('_', ' ').title()] = f"{data[col].sum():,.0f}"
    
    except Exception as e:
        kpis['Error'] = str(e)
    
    return kpis

def generate_persona_recommendations(data: pd.DataFrame, persona_type: PersonaType) -> List[str]:
    """Generate persona-specific recommendations"""
    
    if data.empty:
        return ["No data available for recommendations"]
    
    recommendations = []
    
    try:
        if persona_type == PersonaType.SALES_MANAGER:
            # Sales-specific recommendations
            if 'total_revenue' in data.columns:
                avg_revenue = data['total_revenue'].mean()
                recommendations.append(f"Average revenue per territory: ${avg_revenue:,.2f}")
            if 'Territory' in data.columns:
                territory_count = data['Territory'].nunique()
                recommendations.append(f"Focus on top-performing territories among {territory_count} regions")
        
        elif persona_type == PersonaType.PRODUCT_ANALYST:
            # Product-specific recommendations  
            if 'profit' in data.columns or any('profit' in col.lower() for col in data.columns):
                recommendations.append("Review products with negative margins for pricing optimization")
            if 'ProductName' in data.columns:
                product_count = data['ProductName'].nunique()
                recommendations.append(f"Analyze performance across {product_count} products")
        
        elif persona_type == PersonaType.CUSTOMER_SUCCESS:
            # Customer-specific recommendations
            if 'CustomerID' in data.columns or 'FullName' in data.columns:
                customer_col = 'CustomerID' if 'CustomerID' in data.columns else 'FullName'
                customer_count = data[customer_col].nunique()
                recommendations.append(f"Monitor relationship health for {customer_count} customers")
            recommendations.append("Focus on increasing customer lifetime value and retention")
        
        elif persona_type == PersonaType.EXECUTIVE:
            # Executive-specific recommendations
            recommendations.extend([
                "Monitor key performance trends for strategic planning",
                "Consider resource allocation based on performance data", 
                "Review market opportunities and competitive positioning"
            ])
    
    except Exception as e:
        recommendations = [f"Error generating recommendations: {str(e)}"]
    
    return recommendations[:3]  # Limit to top 3 recommendations

# =============================================================================
# WORKFLOW CONTROL FUNCTIONS
# =============================================================================

def determine_next_node(state: AgentState) -> str:
    """Determine the next node based on the current state"""
    next_action = state.get("next_action", "handle_error")
    
    routing_map = {
        "generate_sql": "sql_generator",
        "execute_sql": "sql_executor", 
        "generate_insights": "insights_generator",
        "format_response": "response_formatter",
        "handle_error": "error_handler",
        "request_clarification": "clarification_handler",
        "generate_empty_response": "response_formatter",
        "complete": END
    }
    
    return routing_map.get(next_action, "error_handler")

# =============================================================================
# MAIN MULTI-AGENT SYSTEM CLASS
# =============================================================================

class PersonaMultiAgentSystem:
    """Main multi-agent system using current LangGraph API"""
    
    def __init__(self):
        # Initialize the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow using current API"""
        
        # Create the StateGraph with our state definition
        workflow = StateGraph(AgentState)
        
        # Add all the agent nodes
        workflow.add_node("persona_router", persona_router_node)
        workflow.add_node("sql_generator", sql_generator_node)
        workflow.add_node("sql_executor", sql_executor_node)
        workflow.add_node("insights_generator", insights_generator_node)
        workflow.add_node("response_formatter", response_formatter_node)
        workflow.add_node("error_handler", error_handler_node)
        workflow.add_node("clarification_handler", clarification_node)
        
        # Set the entry point
        workflow.add_edge(START, "persona_router")
        
        # Add conditional edges based on next_action
        workflow.add_conditional_edges(
            "persona_router",
            determine_next_node
        )
        
        workflow.add_conditional_edges(
            "sql_generator", 
            determine_next_node
        )
        
        workflow.add_conditional_edges(
            "sql_executor",
            determine_next_node
        )
        
        workflow.add_conditional_edges(
            "insights_generator",
            determine_next_node
        )
        
        workflow.add_conditional_edges(
            "response_formatter",
            determine_next_node
        )
        
        # Error handlers and clarification go to END
        workflow.add_edge("error_handler", END)
        workflow.add_edge("clarification_handler", END)
        
        # Compile the graph
        return workflow.compile()
    
    def process_query(self, user_query: str, selected_persona: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        
        # Initialize state
        initial_state = AgentState(
            user_query=user_query,
            selected_persona=selected_persona,
            detected_persona="",
            confidence_score=0.0,
            sql_query="",
            data_results=None,
            insights=[],
            recommendations=[],
            kpis={},
            next_action="",
            error_message=None,
            messages=[]
        )
        
        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)
            
            # Return results
            return {
                "success": True,
                "messages": final_state.get("messages", []),
                "persona": final_state.get("detected_persona", "unknown"),
                "confidence": final_state.get("confidence_score", 0.0),
                "sql_query": final_state.get("sql_query", ""),
                "data_results": final_state.get("data_results"),
                "insights": final_state.get("insights", []),
                "kpis": final_state.get("kpis", {}),
                "recommendations": final_state.get("recommendations", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "messages": [AIMessage(content=f"❌ System error: {str(e)}")],
                "persona": "unknown",
                "confidence": 0.0,
                "sql_query": "",
                "data_results": None,
                "insights": [],
                "kpis": {},
                "recommendations": []
            }

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def create_langgraph_persona_dashboard():
    """Create dashboard with updated LangGraph integration"""
    
    st.set_page_config(
        page_title="🎯 LangGraph Persona Dashboard",
        page_icon="🎯",
        layout="wide"
    )
    
    # Initialize system
    if 'multi_agent_system' not in st.session_state:
        with st.spinner("Initializing LangGraph Multi-Agent System..."):
            st.session_state.multi_agent_system = PersonaMultiAgentSystem()
    
    system = st.session_state.multi_agent_system
    
    # Header
    st.title("🎯 LangGraph Persona Dashboard")
    st.markdown("*Multi-Agent Orchestration with Your Adventure Works Data*")
    
    # Show available data metrics
    col1, col2, col3, col4 = st.columns(4)
    try:
        with col1:
            st.metric("Customers", len(customers))
        with col2:
            st.metric("Products", len(products))
        with col3:
            st.metric("Sales Records", len(sales))
        with col4:
            st.metric("Employees", len(employee))
    except:
        st.info("Data metrics will be shown when model.py is properly imported")
    
    # Persona selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎭 Select Persona")
        persona_options = {
            "sales_manager": "📈 Sales Manager",
            "product_analyst": "📊 Product Analyst",
            "customer_success": "👥 Customer Success",
            "executive": "🎯 Executive"
        }
        
        selected_persona = st.selectbox(
            "Choose your role:",
            options=list(persona_options.keys()),
            format_func=lambda x: persona_options[x],
            key="persona_select"
        )
        
        # Show persona info
        if selected_persona in [p.value for p in PersonaType if p != PersonaType.UNKNOWN]:
            persona_type = PersonaType(selected_persona)
            config = PERSONA_CONFIGS[persona_type]
            st.markdown(f"**{config['icon']} {config['name']}**")
            st.markdown(f"*Focus:* {config['focus']}")
    
    with col2:
        st.subheader("💬 LangGraph Analysis")
        
        # Query input
        user_query = st.text_input(
            "Enter your business question:",
            placeholder="e.g., What's our sales performance by territory?"
        )
        
        if st.button("🚀 Process with LangGraph", type="primary"):
            if user_query:
                with st.spinner("Processing through LangGraph multi-agent workflow..."):
                    # Process query through LangGraph
                    result = system.process_query(user_query, selected_persona)
                    
                    # Display results
                    if result["success"]:
                        st.success(f"✅ Analysis completed by {result['persona']} agents")
                        
                        # Display KPIs
                        if result['kpis']:
                            st.subheader("📈 Key Performance Indicators")
                            kpi_cols = st.columns(len(result['kpis']))
                            for i, (kpi_name, kpi_value) in enumerate(result['kpis'].items()):
                                with kpi_cols[i]:
                                    st.metric(kpi_name, kpi_value)
                        
                        # Display messages
                        for msg in result["messages"]:
                            if isinstance(msg, AIMessage):
                                st.markdown(msg.content)
                        
                        # Display additional details
                        if result["sql_query"]:
                            with st.expander("🔍 Generated SQL"):
                                st.code(result["sql_query"], language="sql")
                        
                        if result["data_results"] is not None and not result["data_results"].empty:
                            with st.expander("📊 Data Results"):
                                st.dataframe(result["data_results"], use_container_width=True)
                        
                        # Show recommendations
                        if result["recommendations"]:
                            with st.expander("💡 Recommendations"):
                                for i, rec in enumerate(result["recommendations"], 1):
                                    st.write(f"{i}. {rec}")
                    
                    else:
                        st.error("❌ Analysis failed")
                        if result.get("messages"):
                            for msg in result["messages"]:
                                if isinstance(msg, AIMessage):
                                    st.write(msg.content)
                        else:
                            st.write(f"Error: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a query")
    
    # Workflow visualization
    with st.expander("🔄 LangGraph Workflow"):
        st.markdown("""
        **Multi-Agent Workflow:**
        1. **Persona Router** → Detects or uses selected persona
        2. **SQL Generator** → Creates persona-enhanced SQL using your model.py
        3. **SQL Executor** → Runs query on your Adventure Works data
        4. **Insights Generator** → Creates persona-specific insights
        5. **Response Formatter** → Formats final response for the persona
        
        *Error handling and clarification requests are handled at each step.*
        """)

if __name__ == "__main__":
    create_langgraph_persona_dashboard()