"""
Module OpenAI - Provider pour les modèles OpenAI
=================================================
Implémentation spécifique pour gpt-4.1 avec gestion de max_completion_tokens
et support des batches.
"""

from typing import List, Dict, Any
from .core import BaseProvider, APIError
from .batch import OpenAIBatchMixin, BatchRequest

# Import de la bibliothèque OpenAI
try:
    import openai
except ImportError:
    openai = None
    print("Attention: bibliothèque openai non installée")


# =============================================================================
# Provider OpenAI avec support batch
# =============================================================================

class OpenAIProvider(OpenAIBatchMixin, BaseProvider):
    """Provider pour le modèle OpenAI gpt-4.1 avec support des batches"""
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialise le client OpenAI.
        
        Args:
            model_name: Nom du modèle OpenAI (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
            api_key: Clé API OpenAI
        """
        super().__init__(model_name, api_key)
        
        if openai is None:
            raise ImportError("Installez openai: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def _filtrer_parametres_openai(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filtre les paramètres pour ne garder que ceux supportés par OpenAI.
        
        Args:
            params: Paramètres à filtrer
            
        Returns:
            dict: Paramètres filtrés
        """
        # Liste des paramètres supportés par OpenAI
        parametres_supportes = {
            'temperature', 'top_p', 'n', 'stream', 'stop',
            'max_tokens', 'max_completion_tokens',
            'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'seed',
            'response_format', 'tools', 'tool_choice'
        }
        
        # Filtrer pour ne garder que les paramètres supportés
        return {k: v for k, v in params.items() if k in parametres_supportes}

    def preparer_parametres_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare et filtre les paramètres pour une requête batch OpenAI."""
        prepared_params = self._preparer_parametres(**params)
        filtered_params = self._filtrer_parametres_openai(prepared_params)
        if 'max_tokens' in filtered_params and self.model_name.startswith('gpt-4.1'):
            filtered_params['max_completion_tokens'] = filtered_params.pop('max_tokens')
        return filtered_params
    
    def generer_reponse(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse avec l'API OpenAI.
        
        Args:
            prompt: Le prompt utilisateur
            **kwargs: Paramètres OpenAI (temperature, max_tokens, etc.)
            
        Returns:
            str: La réponse générée
        """
        if not prompt:
            raise ValueError("Le prompt ne peut pas être vide")
        
        params = self._preparer_parametres(**kwargs)
        
        # Filtrer les paramètres non supportés (comme top_k)
        params = self._filtrer_parametres_openai(params)
        
        # GPT-4.1 famille utilise max_completion_tokens au lieu de max_tokens
        if 'max_tokens' in params and self.model_name.startswith('gpt-4.1'):
            params['max_completion_tokens'] = params.pop('max_tokens')
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise APIError(f"Erreur OpenAI: {str(e)}")
    
    def chatter(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Gère une conversation avec l'API OpenAI.
        
        Args:
            messages: Historique de la conversation
            **kwargs: Paramètres OpenAI
            
        Returns:
            str: La réponse de l'assistant
        """
        if not messages:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        # Valider le format des messages
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Chaque message doit avoir 'role' et 'content'")
        
        params = self._preparer_parametres(**kwargs)
        
        # Filtrer les paramètres non supportés (comme top_k)
        params = self._filtrer_parametres_openai(params)
        
        # GPT-4.1 famille utilise max_completion_tokens
        if 'max_tokens' in params and self.model_name.startswith('gpt-4.1'):
            params['max_completion_tokens'] = params.pop('max_tokens')
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise APIError(f"Erreur OpenAI: {str(e)}")
    
    # La méthode submit_batch est héritée du mixin OpenAIBatchMixin
    # Elle utilise self.client qui est initialisé dans __init__