import streamlit as st
import os
import json
from Agent import agent_executor, parser  # Importing the agent from Agent.py

st.title("ThesiUs - Your AI Research Co-Author")
st.markdown("Ask any research-related question, and ThesiUs will assist you!")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def get_response(query: str):
    """Invoke the agent and return structured response."""
    response = agent_executor.invoke({"query": query})
    try:
        structured_response = parser.parse(response.get("output")[0]["text"])
        return structured_response
    except Exception as e:
        return {"error": f"Parsing error: {e}", "raw_response": response}

# User input field
user_query = st.text_input("Enter your research query:", key="user_input")

if st.button("Submit") and user_query:
    with st.spinner("Generating response..."):
        result = get_response(user_query)
    
    if "error" in result:
        st.error(result["error"])
    else:
        # Store response in conversation history
        st.session_state.conversation.append((user_query, result))
        
        # Display structured response
        st.subheader("Research Summary")
        st.write(result.summary)

        st.subheader("Sources")
        for source in result.sources:
            st.write(f"- {source}")

        st.subheader("Tools Used")
        st.write(", ".join(result.tools_used))

# Show conversation history
if st.session_state.conversation:
    st.markdown("### Previous Research Queries")
    for query, response in st.session_state.conversation:
        st.markdown(f"**Query:** {query}")
        st.markdown(f"**Summary:** {response.summary}")
        st.markdown(f"**Sources:** {', '.join(response.sources)}")
        st.markdown(f"**Tools Used:** {', '.join(response.tools_used)}")
        st.markdown("---")  # Separator

st.markdown("Powered by **ChatGroq** and LangChain")
