import json
import re

def extract_reformulated_prompts(json_data):
    """
    Extracts all "Prompt reformulé" values from a JSON structure.
    
    Args:
        json_data (str): JSON data as a string, potentially embedded in text/markdown
        
    Returns:
        list: List of all reformulated prompts
    """
    # Si json_data est vide ou None, retourner une liste vide
    if not json_data:
        print("Données JSON vides")
        return []
        
    try:
        # 1. Tenter d'extraire le JSON à partir d'un bloc de code markdown ```json ... ```
        json_code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_data, re.DOTALL)
        if json_code_block:
            potential_json = json_code_block.group(1).strip()
            try:
                data = json.loads(potential_json)
                print("JSON extrait d'un bloc de code markdown")
            except json.JSONDecodeError:
                # Si le bloc de code n'est pas un JSON valide, continuer avec d'autres méthodes
                pass
        
        # 2. Essayer de trouver tout objet JSON entre accolades
        if 'data' not in locals():
            json_pattern = r'({.*})'
            json_matches = re.findall(json_pattern, json_data, re.DOTALL)
            
            if json_matches:
                # Essayer chaque correspondance potentielle
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        print("JSON extrait d'une correspondance d'accolades")
                        break
                    except json.JSONDecodeError:
                        continue
        
        # 3. Dernier recours: essayer de parser le texte entier comme JSON
        if 'data' not in locals():
            try:
                data = json.loads(json_data)
                print("JSON parsé directement")
            except json.JSONDecodeError as e:
                print(f"Impossible de parser les données comme JSON: {e}")
                print(f"Début des données: {json_data[:200]}...")
                return []
                
        # Vérifier si "requêtes" est présent (format différent)
        if "requêtes" in data:
            # Format spécifique pour Divider_prompt
            return data["requêtes"]
                
        # Vérifier si le JSON contient "Sous-sujets"
        if "Sous-sujets" not in data:
            print("Structure JSON non reconnue. Clés disponibles:", list(data.keys()))
            return []
        
        # Extraire tous les "Prompt reformulé"
        prompts = []
        for sous_sujet in data["Sous-sujets"]:
            if "Prompt reformulé" in sous_sujet:
                prompts.append(sous_sujet["Prompt reformulé"])
        
        return prompts
    
    except Exception as e:
        print(f"Erreur lors de l'extraction des prompts: {type(e).__name__}: {e}")
        print(f"Début des données: {json_data[:200]}...")
        return []


# Example usage:
if __name__ == "__main__":
    # Test avec JSON standard
    example_json = '''
    {
      "Instruction originale": "Fait un etude de marceh sur la viabilite des telephones solide dans le marceh europeun",
      "Découpe nécessaire": "Oui",
      "Sous-sujets": [
        {
          "Nom du sous-sujet": "Analyse de la demande pour les téléphones solides en Europe",
          "Prompt reformulé": "Analyse la demande actuelle et potentielle pour les téléphones dits 'solides' (résistants aux chocs, à l'eau, etc.) sur le marché européen : qui sont les utilisateurs cibles, quels sont leurs besoins spécifiques, quelles tendances peut-on observer ?"
        },
        {
          "Nom du sous-sujet": "Analyse de l'offre existante et de la concurrence",
          "Prompt reformulé": "Identifie les principaux acteurs proposant des téléphones solides en Europe, leurs caractéristiques produits, gammes de prix, stratégies marketing et parts de marché."
        }
      ]
    }
    '''
    
    # Test avec JSON dans un bloc de code markdown
    markdown_json = '''
    Voici ma réponse:
    
    ```json
    {
      "requêtes": [
        "TVA definition",
        "current TVA rate 2023",
        "implementation of TVA across countries",
        "impact of TVA on imports and exports"
      ]
    }
    ```
    '''
    
    prompts1 = extract_reformulated_prompts(example_json)
    print("\nPrompts extraits du JSON standard:")
    for i, prompt in enumerate(prompts1, 1):
        print(f"{i}. {prompt}")
        
    prompts2 = extract_reformulated_prompts(markdown_json)
    print("\nPrompts extraits du markdown:")
    for i, prompt in enumerate(prompts2, 1):
        print(f"{i}. {prompt}") 