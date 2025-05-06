from test_ollama import test_ollama_chain, test_ollama_chain_with_params
from fisc_gpt import FiscGPT

# Créer une instance de FiscGPT pour obtenir l'entrée
ask = FiscGPT()

# Utiliser différents prompts
# Prompt par défaut
result_default = test_ollama_chain(ask)

# Prompt de programmeur
result_coder = test_ollama_chain(ask, "coder")

# Prompt de traduction (avec paramètre de langue)
result_translator = test_ollama_chain_with_params(ask, "translator", language="français")

# Prompt de résumé
result_summarizer = test_ollama_chain(ask, "summarizer")

print("Exemple d'utilisation terminé!") 