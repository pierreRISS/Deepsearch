from test_ollama import test_ollama_chain, init_ollama, get_json_response
from fisc_gpt import FiscGPT
from extract_prompts import extract_reformulated_prompts
# from text_to_markdown import summarize_and_convert_to_md, save_markdown_to_file
from test_searcher import process_ai_generated_questions
from SeleniumExtractor import SeleniumExtractor
from GoogleSearcher import GoogleSearcher
from vector_db import process_markdown_content
import json
import ast



if __name__ == "__main__":
    print("Testing Ollama with LangChain...")
    extractor = SeleniumExtractor(headless=True, wait_time=5, scroll_page=True)
    init_ollama()
    google_searcher = GoogleSearcher()
    ask = FiscGPT()
    # Test with default prompt
    # sub_sujets = get_json_response(ask, "divider_subject")

    # Utilisation correcte d'un dictionnaire Python
    sub_sujets_str = """{'Instruction originale': 'Tva sas ?', 'Découpe nécessaire': 'Oui', 'Sous-sujets': [{'Nom du sous-sujet': 'Analyse de la demande pour les téléphones solides en Europe', 'Prompt reformulé': "Analyse la demande actuelle et potentielle pour les téléphones dits 'solides' (résistants aux chocs, à l'eau, etc.) sur le marché européen : qui sont les utilisateurs cibles, quels sont leurs besoins spécifiques, quelles tendances peut-on observer ?"}, {'Nom du sous-sujet': "Analyse de l'offre existante et de la concurrence", 'Prompt reformulé': 'Identifie les principaux acteurs proposant des téléphones solides en Europe, leurs caractéristiques produits, gammes de prix, stratégies marketing et parts de marché.'}]}"""
    sub_sujets = ast.literal_eval(sub_sujets_str)
    
    print("sub_sujets", sub_sujets)
    # sub_sujets = sub_sujets["requêtes"]
    # Sous-sujets
    for sub_sujet in sub_sujets["Sous-sujets"]:
        requettes = get_json_response(sub_sujet["Prompt reformulé"], "Divider_prompt")["requêtes"]
        print("requettes", requettes)
        urls = google_searcher.batch_search(requettes)
        print("urls", urls)
        markdown_contents = extractor.process_urls(urls)
        process_markdown_content(markdown_contents, markdown_contents)
        # prompts = extract_reformulated_prompts(test_ollama_chain(ask, "divider_subject"))
        # query_search = test_ollama_chain(ask, "Divider_prompt")

    # for i in range(len(requettes)):
    #     questions = test_ollama_chain(ask, requettes)
    # print(process_ai_generated_questions(ouptut["requêtes"]))
    
    # md_summary = summarize_and_convert_to_md(pages_exemple)
    # print(md_summary)
    
    # # Sauvegarder le markdown dans un fichier
    # save_markdown_to_file(md_summary)
    # # Test with coder prompt if needed
    # # test_ollama_chain(ask, "coder")
    
    print("Test completed!") 