"""
Module GPT-5 - Provider pour les modèles GPT-5
===============================================
Implémentation spécifique pour la famille GPT-5 avec reasoning_effort, verbosity
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
# Provider GPT-5 avec support batch
# =============================================================================

class GPT5Provider(OpenAIBatchMixin, BaseProvider):
    """
    Provider spécifique pour GPT-5 avec ses paramètres uniques et support des batches.
    GPT-5 introduit reasoning_effort et verbosity à la place de temperature/top_p.
    """
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialise le client pour GPT-5.
        
        Args:
            model_name: Nom du modèle GPT-5 (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat-latest)
            api_key: Clé API OpenAI
        """
        super().__init__(model_name, api_key)
        
        if openai is None:
            raise ImportError("Installez openai: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Paramètres par défaut spécifiques à GPT-5
        self.gpt5_defaults = {
            'reasoning_effort': 'medium',  # minimal, low, medium, high
            'verbosity': 'medium'  # low, medium, high
        }
    
    def _preparer_parametres_gpt5(self, **kwargs) -> Dict[str, Any]:
        """
        Prépare les paramètres spécifiques à GPT-5.
        
        Args:
            **kwargs: Paramètres utilisateur
            
        Returns:
            dict: Paramètres formatés pour GPT-5
        """
        # Initialiser les paramètres
        params = {}
        
        # GPT-5 utilise max_completion_tokens (comme GPT-4.1), pas max_tokens
        if 'max_tokens' in kwargs:
            params['max_completion_tokens'] = kwargs['max_tokens']
        else:
            params['max_completion_tokens'] = self.default_params.get('max_tokens', 1000)
        
        # Ajouter les paramètres spécifiques à GPT-5
        params['reasoning_effort'] = kwargs.get('reasoning_effort', self.gpt5_defaults['reasoning_effort'])
        params['verbosity'] = kwargs.get('verbosity', self.gpt5_defaults['verbosity'])

        # Forcer le raisonnement minimal pour gpt-5-nano
        if self.model_name == 'gpt-5-nano':
            params['reasoning_effort'] = 'minimal'
        
        # En mode minimal, on peut utiliser temperature et autres paramètres classiques
        if params['reasoning_effort'] == 'minimal':
            # N'ajouter ces paramètres que si le modèle n'est pas gpt-5-nano
            if self.model_name != 'gpt-5-nano':
                if 'temperature' in kwargs:
                    params['temperature'] = kwargs['temperature']
                if 'top_p' in kwargs:
                    params['top_p'] = kwargs['top_p']
                if 'frequency_penalty' in kwargs:
                    params['frequency_penalty'] = kwargs['frequency_penalty']
                if 'presence_penalty' in kwargs:
                    params['presence_penalty'] = kwargs['presence_penalty']
        # Sinon, en mode raisonnement (low, medium, high), ces paramètres sont ignorés
        
        # Filtrer les paramètres None
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def preparer_parametres_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare et filtre les paramètres pour une requête batch GPT-5."""
        return self._preparer_parametres_gpt5(**params)
    
    def generer_reponse(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse avec GPT-5.
        
        Args:
            prompt: Le prompt utilisateur
            **kwargs: Paramètres incluant reasoning_effort et verbosity
            
        Returns:
            str: La réponse générée
        """
        if not prompt:
            raise ValueError("Le prompt ne peut pas être vide")
        
        params = self._preparer_parametres_gpt5(**kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Utiliser l'API avec les paramètres GPT-5
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            
            # Gérer les erreurs spécifiques
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                # Cette erreur ne devrait plus arriver avec notre correction
                raise APIError(f"Erreur de paramètre GPT-5: utilisez max_tokens dans l'appel, il sera converti en max_completion_tokens automatiquement")
            
            # Si les paramètres reasoning_effort/verbosity ne sont pas supportés (fallback pour compatibilité)
            if "reasoning_effort" in error_str or "verbosity" in error_str:
                # Certains modèles pourraient ne pas supporter ces paramètres
                # Essayer sans ces paramètres
                params_fallback = {
                    'max_completion_tokens': params.get('max_completion_tokens', 1000)
                }

                # Si on était en mode minimal, ajouter les paramètres classiques
                if params.get('reasoning_effort') == 'minimal' and self.model_name != 'gpt-5-nano':
                    for key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                        if key in params:
                            params_fallback[key] = params[key]
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **params_fallback
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    raise APIError(f"Erreur GPT-5 (fallback): {str(e2)}")
            
            raise APIError(f"Erreur GPT-5: {error_str}")
    
    def chatter(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Gère une conversation avec GPT-5.
        
        Args:
            messages: Historique de la conversation
            **kwargs: Paramètres incluant reasoning_effort et verbosity
            
        Returns:
            str: La réponse de l'assistant
        """
        if not messages:
            raise ValueError("La liste de messages ne peut pas être vide")
        
        # Valider le format des messages
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Chaque message doit avoir 'role' et 'content'")
        
        params = self._preparer_parametres_gpt5(**kwargs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            
            # Gérer les erreurs de paramètres
            if "max_tokens" in error_str and "max_completion_tokens" in error_str:
                raise APIError(f"Erreur de paramètre GPT-5: utilisez max_tokens dans l'appel, il sera converti en max_completion_tokens automatiquement")
            
            # Fallback sans paramètres GPT-5 si nécessaire
            if "reasoning_effort" in error_str or "verbosity" in error_str:
                params_fallback = {
                    'max_completion_tokens': params.get('max_completion_tokens', 1000)
                }

                # Si on était en mode minimal, ajouter les paramètres classiques
                if params.get('reasoning_effort') == 'minimal' and self.model_name != 'gpt-5-nano':
                    for key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                        if key in params:
                            params_fallback[key] = params[key]
                
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **params_fallback
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    raise APIError(f"Erreur GPT-5 (fallback): {str(e2)}")
            
            raise APIError(f"Erreur GPT-5: {error_str}")
    
    # La méthode submit_batch est héritée du mixin OpenAIBatchMixin
    # Elle utilise self.client qui est initialisé dans __init__