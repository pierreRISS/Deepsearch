from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from typing import List, Optional, Dict, Any
from fisc_gpt import FiscGPT
from prompts import PROMPTS


# 1. Définir les schémas de JSON attendus
class SearchQueries(BaseModel):
    """Structure pour les requêtes de recherche"""
    requêtes: List[str] = Field(description="Liste des requêtes de recherche Google")

class SubjectDecomposition(BaseModel):
    """Structure pour la décomposition de sujets"""
    instruction_originale: Optional[str] = Field(None, description="L'instruction originale de l'utilisateur")
    découpe_nécessaire: Optional[str] = Field(None, description="Si la découpe est nécessaire (Oui/Non)")
    sous_sujets: Optional[List[Dict[str, Any]]] = Field(None, description="Liste des sous-sujets identifiés")


def init_ollama():
    # Initialize the Ollama model with smollm
    global llm, prompt_templates
    llm = Ollama(model="deepseek-r1:8b")

    # Create prompt templates from the imported prompts
    prompt_templates = {
        name: PromptTemplate.from_template(
            template_text + "\n\nImportant: Ta réponse doit être uniquement un objet JSON valide, sans texte avant ou après."
        ) 
        for name, template_text in PROMPTS.items()
    }

# Create a simple chain using LangChain Expression Language (LCEL)
def create_chain(prompt_name: str, use_json_parser=False):
    selected_prompt = prompt_templates.get(prompt_name)
    
    # Check which variable name is needed based on the prompt name
    if prompt_name == "divider_subject":
        input_key = "question"
    else:
        input_key = "question"
    
    # Select output parser based on parameter
    if use_json_parser:
        if prompt_name == "Divider_prompt":
            output_parser = JsonOutputParser(pydantic_object=SearchQueries)
        else:  # divider_subject
            output_parser = JsonOutputParser(pydantic_object=SubjectDecomposition)
    else:
        output_parser = StrOutputParser()
    
    return (
        {input_key: RunnablePassthrough()} 
        | selected_prompt 
        | llm 
        | output_parser
    )

# Test the chain with verbose output
def test_ollama_chain(prompt_text: str, prompt_name:str, verbose=True, use_json_parser=False):
    chain = create_chain(prompt_name, use_json_parser)
    response = chain.invoke(prompt_text)
    
    if verbose and not use_json_parser:
        print(f"Response using '{prompt_name}' template: {response}")
    elif verbose and use_json_parser:
        print(f"JSON response using '{prompt_name}' template:")
        if prompt_name == "Divider_prompt" and "requêtes" in response:
            for i, query in enumerate(response["requêtes"], 1):
                print(f"{i}. {query}")
        elif "sous_sujets" in response and response["sous_sujets"]:
            for i, sujet in enumerate(response["sous_sujets"], 1):
                print(f"\n{i}. {sujet.get('Nom du sous-sujet', '')}")
                print(f"   Prompt: {sujet.get('Prompt reformulé', '')}")
    
    return response

# Get clean response only (without printing)
def get_clean_response(prompt_text: str, prompt_name:str):
    """
    Gets only the model's response without any additional output
    """
    return test_ollama_chain(prompt_text, prompt_name, verbose=False)

# Get JSON response directly
def get_json_response(prompt_text: str, prompt_name:str, verbose=True):
    """
    Gets the response as a parsed JSON object
    """
    return test_ollama_chain(prompt_text, prompt_name, verbose=verbose, use_json_parser=True)


if __name__ == "__main__":
    print("Testing Ollama with LangChain...")
    ask = FiscGPT()
    
    # Initialiser Ollama
    init_ollama()
    
    print("\n=== Test avec réponse textuelle standard ===")
    test_ollama_chain(ask, "divider_subject")
    
    print("\n=== Test avec réponse JSON parsée - Divider_prompt ===")
    json_response1 = get_json_response(ask, "Divider_prompt")
    
    print("\n=== Test avec réponse JSON parsée - divider_subject ===")
    json_response2 = get_json_response(ask, "divider_subject")
    
    print("\nTest completed!") 