from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fisc_gpt import FiscGPT

# Initialize the Ollama model with smollm
llm = Ollama(model="smollm:latest")

# Create a simple prompt template
prompt = PromptTemplate.from_template(
    "Tu es un assistant fiscal français. Réponds à la question suivante: {question}"
)

# Create a simple chain using LangChain Expression Language (LCEL)
chain = (
    {"question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

def main():
    print("Bienvenue sur FiscGPT! Posez votre question fiscale.")
    
    # Get user input using FiscGPT function
    user_question = FiscGPT()
    
    # Process through Ollama
    response = chain.invoke(user_question)
    
    # Display the response
    print("\nRéponse:")
    print(response)

if __name__ == "__main__":
    main() 