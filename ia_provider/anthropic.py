"""
Module Anthropic - Provider pour les modèles Claude
====================================================
Implémentation spécifique pour claude-sonnet-4 avec support du mode thinking
et des batches.
"""

from typing import List, Dict, Any
from .core import BaseProvider, APIError
from .batch import AnthropicBatchMixin, BatchRequest

# Import de la bibliothèque Anthropic
try:
    import anthropic
except ImportError:
    anthropic = None
    print("Attention: bibliothèque anthropic non installée")


# =============================================================================
# Provider Anthropic avec support batch
# =============================================================================

class AnthropicProvider(AnthropicBatchMixin, BaseProvider):
    """Provider pour le modèle Anthropic claude-sonnet-4 avec support des batches"""
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialise le client Anthropic.
        
        Args:
            model_name: Nom du modèle Claude (claude-sonnet-4)
            api_key: Clé API Anthropic
        """
        # Claude 4 (Sonnet 4) requiert le nom de modèle exact : claude-sonnet-4-20250514
        super().__init__(model_name, api_key)
        
        if anthropic is None:
            raise ImportError("Installez anthropic: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _preparer_parametres_anthropic(self, **kwargs) -> Dict[str, Any]:
        """
        Prépare les paramètres spécifiques à Anthropic.
        
        Args:
            **kwargs: Paramètres utilisateur
            
        Returns:
            dict: Paramètres formatés pour Anthropic
        """
        params = self._preparer_parametres(**kwargs)
        
        # Filtrer les paramètres non supportés par Anthropic
        parametres_non_supportes = {'top_k', 'frequency_penalty', 'presence_penalty', 'seed'}
        params = {k: v for k, v in params.items() if k not in parametres_non_supportes}
        
        # Anthropic requiert max_tokens
        if 'max_tokens' not in params:
            params['max_tokens'] = 1000
        
        # Gestion de stop_sequences vs stop
        if 'stop' in params:
            params['stop_sequences'] = params.pop('stop')
        
        # Pour Claude 4, gérer le paramètre thinking
        if 'thinking_budget' in kwargs:
            params['thinking'] = {
                'type': 'enabled',
                'budget_tokens': kwargs['thinking_budget']
            }
        elif 'thinking' in kwargs:
            params['thinking'] = kwargs['thinking']

        return params

    def preparer_parametres_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare et filtre les paramètres pour une requête batch Anthropic."""
        return self._preparer_parametres_anthropic(**params)
    
    def generer_reponse(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse avec l'API Anthropic.
        
        Args:
            prompt: Le prompt utilisateur
            **kwargs: Paramètres Anthropic (peut inclure thinking_budget)
            
        Returns:
            str: La réponse générée
        """
        if not prompt:
            raise ValueError("Le prompt ne peut pas être vide")
        
        params = self._preparer_parametres_anthropic(**kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.content[0].text
        except Exception as e:
            raise APIError(f"Erreur Anthropic: {str(e)}")
    
    def chatter(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Gère une conversation avec l'API Anthropic.
        
        Args:
            messages: Historique de la conversation
            **kwargs: Paramètres Anthropic (peut inclure thinking_budget)
            
        Returns:
            str: La réponse de l'assistant
        """
        if not messages:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        # Valider le format des messages
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Chaque message doit avoir 'role' et 'content'")
            if msg['role'] not in ['user', 'assistant']:
                raise ValueError("Les rôles doivent être 'user' ou 'assistant'")
        
        params = self._preparer_parametres_anthropic(**kwargs)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.content[0].text
        except Exception as e:
            raise APIError(f"Erreur Anthropic: {str(e)}")
    
    # La méthode submit_batch est héritée du mixin AnthropicBatchMixin
    # Elle utilise self.client qui est initialisé dans __init__