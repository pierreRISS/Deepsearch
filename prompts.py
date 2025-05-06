"""
This file stores prompt templates for use with LangChain.
"""

PROMPTS = {
    "Divider_prompt": """
Tu es un assistant intelligent spécialisé dans la recherche d'information.
À partir d'une question ou instruction utilisateur, génère un JSON contenant une liste de requêtes Google pertinentes pour aider à répondre à cette demande.
Sois précis, contextuel, et exhaustif, mais ne génère pas de réponses : seulement les recherches nécessaires.

Ton objectif est de décomposer la demande, analyser les sous-questions implicites, et traduire cela en un mot clef efficace. X. Redige le moins mot possible.

Format de sortie (exemple) :
json
Copy
Edit
{{{{
  "requêtes": [
    "définition du machine learning",
    "exemples d'application du machine learning",
    "différences deep learning vs machine learning"
  ]
}}}}
Voici la question ou l'instruction de l'utilisateur : {question}
""",
"divider_subject": """
Tu es un assistant intelligent spécialisé en décomposition de requêtes complexes.
Reçois une question ou instruction utilisateur (dans n'importe quel domaine).

Étapes à suivre :

Analyse en profondeur la question : s'agit-il d'une seule requête ou d'une demande contenant plusieurs sous-sujets logiquement liés mais distincts ?

Si et seulement si nécessaire, découpe la requête en X sous-sujets clairs, larges et hiérarchiquement cohérents (ne jamais découper artificiellement une question simple).

Chaque sous-sujet doit être pertinent, autonome et avoir une portée suffisante pour justifier un traitement séparé.

N'invente pas de sous-sujets : reste fidèle à l'intention utilisateur.

Retourne les sous-sujets identifiés (ou indique qu'aucune découpe n'est nécessaire).

Pour chaque sous-sujet (s'il y en a), reformule un prompt clair et ciblé à utiliser pour obtenir une réponse optimale.

✳️ Format de sortie :
Ecrit un json de sortie

Instruction originale :
{question} 

{{{{
  "Instruction originale": "Fait un etude de marceh sur la viabilite des telephones solide dans le marceh europeun",
  "Découpe nécessaire": "Oui",
  "Sous-sujets": [
    {{{{
      "Nom du sous-sujet": "Analyse de la demande pour les téléphones solides en Europe",
      "Prompt reformulé": "Analyse la demande actuelle et potentielle pour les téléphones dits 'solides' (résistants aux chocs, à l'eau, etc.) sur le marché européen : qui sont les utilisateurs cibles, quels sont leurs besoins spécifiques, quelles tendances peut-on observer ?"
    }}}},
    {{{{
      "Nom du sous-sujet": "Analyse de l'offre existante et de la concurrence",
      "Prompt reformulé": "Identifie les principaux acteurs proposant des téléphones solides en Europe, leurs caractéristiques produits, gammes de prix, stratégies marketing et parts de marché."
    }}}},
    ...
  ]
}}}}

..."""
}

# You can add more prompts as needed 