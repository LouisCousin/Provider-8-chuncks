"""
Module Core - Classes de base et utilitaires
============================================
Contient les classes abstraites, le gestionnaire et les fonctions utilitaires.
"""

import os
import yaml
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type
from dotenv import load_dotenv


# =============================================================================
# Exceptions personnalisées
# =============================================================================

class APIError(Exception):
    """Exception levée lors d'erreurs d'API."""
    pass


class UnknownModelError(ValueError):
    """Exception levée pour un modèle non reconnu."""
    pass


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def load_config() -> Dict[str, Any]:
    """
    Charge la configuration depuis config.yaml ou retourne les défauts.
    
    Returns:
        dict: Configuration avec paramètres par défaut
    """
    config_path = "config.yaml"
    default_config = {
        'temperature': 0.7,
        'max_tokens': 1000,
        'top_p': 0.95,
        'top_k': 40,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'seed': None
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
        except Exception as e:
            print(f"Erreur lors de la lecture de la configuration: {e}")
    
    return default_config


def load_api_key(provider_name: str) -> str:
    """
    Récupère la clé API pour un provider donné depuis l'environnement.
    
    Args:
        provider_name: Nom de la classe du provider
        
    Returns:
        str: Clé API
        
    Raises:
        ValueError: Si la clé n'est pas trouvée
    """
    load_dotenv()
    
    env_mapping = {
        'OpenAIProvider': 'OPENAI_API_KEY',
        'GPT5Provider': 'OPENAI_API_KEY',  # GPT5 utilise aussi OpenAI
        'AnthropicProvider': 'ANTHROPIC_API_KEY',
        'GoogleProvider': 'GOOGLE_API_KEY'
    }
    
    env_var = env_mapping.get(provider_name)
    
    if env_var:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    
    raise ValueError(
        f"Clé API non trouvée pour {provider_name}. "
        f"Veuillez définir {env_var} dans .env ou la fournir explicitement."
    )


# =============================================================================
# Classe de base abstraite
# =============================================================================

class BaseProvider(ABC):
    """
    Classe de base abstraite pour tous les providers IA.
    Définit l'interface commune que tous les providers doivent implémenter.
    """
    
    def __init__(self, model_name: str, api_key: str):
        """
        Initialise le provider avec le modèle et la clé API.
        
        Args:
            model_name: Nom du modèle à utiliser
            api_key: Clé API pour l'authentification
        """
        if not api_key:
            raise ValueError("La clé API ne peut pas être vide")
        
        self.model_name = model_name
        self.api_key = api_key
        self.default_params = load_config()
    
    @abstractmethod
    def generer_reponse(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt simple.
        
        Args:
            prompt: Le texte d'entrée
            **kwargs: Paramètres additionnels (temperature, max_tokens, etc.)
            
        Returns:
            str: La réponse générée
        """
        pass
    
    @abstractmethod
    def chatter(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Gère une conversation multi-tours.
        
        Args:
            messages: Liste de messages avec 'role' et 'content'
            **kwargs: Paramètres additionnels
            
        Returns:
            str: La réponse de l'assistant
        """
        pass
    
    @abstractmethod
    def submit_batch(self, requests: List['BatchRequest'], metadata: Dict = None) -> str:
        """
        Soumet un lot de requêtes et retourne un ID de travail.
        
        Args:
            requests: Liste de BatchRequest à traiter
            metadata: Métadonnées optionnelles pour le batch
            
        Returns:
            str: ID du batch créé
            
        Raises:
            APIError: Si le provider ne supporte pas les batches
        """
        pass

    @abstractmethod
    def preparer_parametres_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prépare et filtre les paramètres pour une requête de type batch."""
        pass
    
    def _preparer_parametres(self, **kwargs) -> Dict[str, Any]:
        """
        Fusionne les paramètres par défaut avec ceux fournis.
        
        Args:
            **kwargs: Paramètres fournis par l'utilisateur
            
        Returns:
            dict: Paramètres fusionnés
        """
        params = self.default_params.copy()
        
        # Filtrer les valeurs None
        kwargs_filtered = {k: v for k, v in kwargs.items() if v is not None}
        params.update(kwargs_filtered)
        
        # Retirer les paramètres None du résultat final
        return {k: v for k, v in params.items() if v is not None}


# =============================================================================
# Gestionnaire central des providers
# =============================================================================

class ProviderManager:
    """
    Gestionnaire central pour tous les providers IA.
    Permet la sélection dynamique et l'enregistrement de nouveaux providers.
    """
    
    def __init__(self):
        """Initialise le gestionnaire."""
        self.providers: Dict[Type[BaseProvider], List[str]] = {}
        self.model_to_provider: Dict[str, Type[BaseProvider]] = {}
        self.config = load_config()
    
    def register_provider(self, provider_class: Type[BaseProvider], models: List[str]):
        """
        Enregistre un nouveau provider avec ses modèles.
        
        Args:
            provider_class: Classe du provider (doit hériter de BaseProvider)
            models: Liste des noms de modèles supportés
            
        Raises:
            TypeError: Si la classe n'hérite pas de BaseProvider
            ValueError: Si aucun modèle n'est fourni
        """
        if not issubclass(provider_class, BaseProvider):
            raise TypeError(f"{provider_class.__name__} doit hériter de BaseProvider")
        
        if not models:
            raise ValueError("Au moins un modèle doit être spécifié")
        
        self.providers[provider_class] = models
        
        # Créer le mapping modèle -> provider
        for model in models:
            if model in self.model_to_provider:
                print(f"Attention: {model} est déjà enregistré, écrasement du provider")
            self.model_to_provider[model] = provider_class
        
        print(f"Provider {provider_class.__name__} enregistré avec {len(models)} modèle(s)")
    
    def get_provider(self, model_name: str, api_key: Optional[str] = None) -> BaseProvider:
        """
        Retourne une instance du provider approprié pour le modèle.
        
        Args:
            model_name: Nom du modèle à utiliser
            api_key: Clé API (optionnel, sera chargée depuis l'environnement si absente)
            
        Returns:
            BaseProvider: Instance du provider configuré
            
        Raises:
            UnknownModelError: Si le modèle n'est pas reconnu
            ValueError: Si la clé API ne peut pas être trouvée
        """
        if model_name not in self.model_to_provider:
            available = self.get_available_models()
            raise UnknownModelError(
                f"Modèle '{model_name}' non reconnu. "
                f"Modèles disponibles: {', '.join(available)}"
            )
        
        provider_class = self.model_to_provider[model_name]
        
        # Charger la clé API si non fournie
        if api_key is None:
            try:
                api_key = load_api_key(provider_class.__name__)
            except ValueError as e:
                raise ValueError(
                    f"Impossible de charger la clé API pour {provider_class.__name__}. "
                    f"Veuillez la fournir explicitement ou la définir dans .env: {e}"
                )
        
        return provider_class(model_name, api_key)
    
    def get_available_models(self) -> List[str]:
        """
        Retourne la liste de tous les modèles disponibles.
        
        Returns:
            List[str]: Liste des noms de modèles
        """
        return sorted(list(self.model_to_provider.keys()))
    
    def get_default_param(self, param_name: str) -> Any:
        """
        Récupère la valeur par défaut d'un paramètre.
        
        Args:
            param_name: Nom du paramètre
            
        Returns:
            Any: Valeur par défaut du paramètre
        """
        return self.config.get(param_name)
    
    def get_providers_info(self) -> Dict[str, List[str]]:
        """
        Retourne les informations sur tous les providers enregistrés.
        
        Returns:
            Dict[str, List[str]]: Mapping provider_name -> liste de modèles
        """
        return {
            provider_class.__name__: models
            for provider_class, models in self.providers.items()
        }