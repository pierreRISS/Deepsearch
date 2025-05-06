import pytest
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@pytest.fixture
def ollama_chain():
    """Fixture to create an Ollama chain for testing."""
    # Initialize the Ollama model
    llm = Ollama(model="smollm:latest")
    
    # Create a simple prompt template
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Answer the following question: {question}"
    )
    
    # Create a simple chain using LangChain Expression Language (LCEL)
    chain = (
        {"question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def test_ollama_response(ollama_chain):
    """Test that the Ollama model returns a non-empty response."""
    response = ollama_chain.invoke("What is the capital of France?")
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    
    print(f"Response: {response}")
    
    return response