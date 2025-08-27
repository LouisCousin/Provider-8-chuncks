"""
Application Streamlit - Interface IA Multi-Modèles
==================================================
Interface utilisateur pour interagir avec différents modèles d'IA
via le module ia_provider unifié.

Version simplifiée avec support pour:
- gpt-4.1 (OpenAI)
- gpt-5 (OpenAI avec reasoning)
- claude-sonnet-4 (Anthropic)
"""

import streamlit as st
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from dataclasses import asdict

# Import du module IA Provider (changement d'import)
try:
    from ia_provider import manager, APIError, UnknownModelError, exporter, importer
    from ia_provider.batch import (
        BatchRequest,
        BatchJobManager,
        BatchResult,
    )
except ImportError:
    st.error("Module ia_provider non trouvé. Assurez-vous que le package ia_provider est dans le même dossier.")
    st.stop()


# =============================================================================
# Configuration de la page
# =============================================================================

st.set_page_config(
    page_title="IA Multi-Modèles",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# Initialisation de la session
# =============================================================================

def init_session_state():
    """Initialise les variables de session."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_mode' not in st.session_state:
        st.session_state.conversation_mode = False
    if 'last_model' not in st.session_state:
        st.session_state.last_model = None
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0
    if 'source_template_styles' not in st.session_state:
        st.session_state.source_template_styles = None


init_session_state()


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def get_model_provider_name(model_name: str) -> str:
    """Retourne le nom du provider pour un modèle donné."""
    if model_name.startswith('gpt'):
        return 'OpenAI'
    elif model_name.startswith('claude'):
        return 'Anthropic'
    return 'Unknown'


def clear_conversation():
    """Efface l'historique de conversation."""
    st.session_state.messages = []
    st.session_state.generation_count = 0


def add_message(role: str, content: str):
    """Ajoute un message à l'historique."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })


def get_api_key(model_name: str) -> Optional[str]:
    """Récupère la clé API pour un modèle."""
    provider = get_model_provider_name(model_name)
    
    # D'abord vérifier dans session state
    if provider in st.session_state.api_keys:
        return st.session_state.api_keys[provider]
    
    # Sinon essayer les variables d'environnement
    env_map = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY'
    }
    
    if provider in env_map:
        return os.getenv(env_map[provider])

    return None


def hex_to_rgb(hex_color: str) -> tuple:
    """Convertit une couleur hexadécimale en tuple RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))



# Dictionnaire des limites de tokens par modèle
MODEL_MAX_TOKENS = {
    "gpt-4.1": 32768,
    "gpt-4.1-mini": 32768,
    "gpt-4.1-nano": 32768,
    "gpt-5": 128000,
    "gpt-5-mini": 128000,
    "gpt-5-nano": 128000,
    "gpt-5-chat-latest": 128000,
    "claude-sonnet-4-20250514": 64000,
}


# =============================================================================
# Interface Sidebar
# =============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Sélection du modèle
    st.subheader("🤖 Modèle")
    available_models = manager.get_available_models()
    
    # Information sur les modèles disponibles
    st.info(f"💡 {len(available_models)} modèle(s) disponible(s)")
    
    selected_model = st.selectbox(
        "Choisissez un modèle",
        options=available_models,
        format_func=lambda x: f"{x} ({get_model_provider_name(x)})",
        index=available_models.index(st.session_state.last_model) if st.session_state.last_model in available_models else 0
    )
    
    st.session_state.last_model = selected_model
    
    # Afficher les capacités spéciales du modèle
    if selected_model.startswith("gpt-4.1"):
        st.caption("✨ Utilise max_completion_tokens")
        if selected_model == "gpt-4.1-nano":
            st.caption("⚡ Modèle ultra-rapide et économique")
        elif selected_model == "gpt-4.1-mini":
            st.caption("⚖️ Équilibre performance/coût")
    elif selected_model.startswith("gpt-5"):
        st.caption("🧠 Utilise reasoning_effort et verbosity")
        if selected_model == "gpt-5-nano":
            st.caption("⚡ Version rapide avec raisonnement minimal")
        elif selected_model == "gpt-5-mini":
            st.caption("⚖️ Raisonnement équilibré")
        elif selected_model == "gpt-5-chat-latest":
            st.caption("💬 Optimisé pour le chat")
        else:
            st.caption("🎯 Raisonnement complet")
    elif selected_model == "claude-sonnet-4-20250514":
        st.caption("✨ Support du mode thinking")
    
    # Clés API
    st.subheader("🔐 Clés API")
    
    provider = get_model_provider_name(selected_model)
    api_key = st.text_input(
        f"Clé API {provider}",
        value=get_api_key(selected_model) or "",
        type="password",
        help=f"Entrez votre clé API {provider} ou définissez-la dans .env"
    )
    
    if api_key:
        st.session_state.api_keys[provider] = api_key
    
    # Paramètres de génération
    st.subheader("🎛️ Paramètres")

    # Récupérer la limite de tokens pour le modèle sélectionné
    max_tokens_for_model = MODEL_MAX_TOKENS.get(selected_model, 4000)

    # Paramètres spécifiques pour GPT-5
    if selected_model.startswith("gpt-5"):
        st.markdown("### 🧠 Paramètres GPT-5")

        # Gérer le cas spécifique de gpt-5-nano
        is_nano = selected_model == "gpt-5-nano"

        if is_nano:
            st.info(
                "💡 Le modèle gpt-5-nano est optimisé pour des réponses rapides et utilise un raisonnement minimal."
            )

        reasoning_effort = st.select_slider(
            "Reasoning Effort",
            options=["minimal", "low", "medium", "high"],
            value="minimal" if is_nano else "medium",
            help="Niveau de raisonnement du modèle",
            disabled=is_nano
        )

        verbosity = st.select_slider(
            "Verbosity",
            options=["low", "medium", "high"],
            value="low" if is_nano else "medium",
            help="Niveau de détail des réponses",
            disabled=is_nano
        )
        
        # Temperature seulement en mode minimal
        if reasoning_effort == "minimal":
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Contrôle la créativité (disponible uniquement en mode minimal)"
            )
        else:
            st.info("💡 Temperature désactivée en mode raisonnement (low/medium/high)")
            temperature = None
    else:
        # Paramètres classiques pour GPT-4.1 et Claude
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=manager.get_default_param('temperature') or 0.7,
            step=0.1,
            help="Contrôle la créativité (0=déterministe, 2=très créatif)"
        )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=max_tokens_for_model,
        value=max_tokens_for_model,
        step=50,
        help="Longueur maximale de la réponse",
    )
    
    # Paramètres spécifiques pour Claude-sonnet-4
    if selected_model == "claude-sonnet-4-20250514":
        st.subheader("🧠 Mode Thinking")
        use_thinking = st.checkbox("Activer le mode thinking", value=False)
        if use_thinking:
            thinking_budget = st.slider(
                "Budget de tokens pour thinking",
                min_value=100,
                max_value=1000,
                value=200,
                step=50,
                help="Nombre de tokens alloués au raisonnement interne"
            )
    
    # Paramètres avancés (sauf pour GPT-5 en mode raisonnement)
    show_advanced = True
    if selected_model.startswith("gpt-5"):
        if 'reasoning_effort' in locals() and reasoning_effort != "minimal":
            show_advanced = False
            st.info("💡 Paramètres avancés désactivés en mode raisonnement")
    
    if show_advanced:
        with st.expander("Paramètres avancés"):
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=manager.get_default_param('top_p') or 0.95,
                step=0.05,
                help="Nucleus sampling"
            )
            
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=0.0,
                max_value=2.0,
                value=manager.get_default_param('frequency_penalty') or 0.0,
                step=0.1,
                help="Pénalité pour la répétition de tokens"
            )
            
            presence_penalty = st.slider(
                "Presence Penalty",
                min_value=0.0,
                max_value=2.0,
                value=manager.get_default_param('presence_penalty') or 0.0,
                step=0.1,
                help="Pénalité pour l'utilisation de nouveaux topics"
            )
    
    # Mode de conversation
    st.subheader("💬 Mode")
    conversation_mode = st.checkbox(
        "Mode Conversation",
        value=st.session_state.conversation_mode,
        help="Active le mode conversation pour garder le contexte"
    )
    st.session_state.conversation_mode = conversation_mode

    st.subheader("🚀 Mode d'exécution")
    execution_mode = st.radio(
        "Choisissez le type de traitement",
        ('Réponse immédiate (Synchrone)', 'Traitement par lot (Batch)'),
        help="Synchrone pour une réponse directe, Batch pour une tâche de fond."
    )

    if conversation_mode and st.button("🗑️ Effacer la conversation"):
        clear_conversation()
        st.rerun()
    
    # Statistiques
    st.subheader("📊 Statistiques")
    st.metric("Générations", st.session_state.generation_count)
    if st.session_state.messages:
        st.metric("Messages", len(st.session_state.messages))

    with st.expander("🎨 Personnaliser l'export DOCX"):
        st.subheader("Style du Prompt")
        prompt_font = st.selectbox(
            "Police (Prompt)",
            ["Arial", "Calibri", "Times New Roman"],
            key="prompt_font",
        )
        prompt_size = st.slider(
            "Taille (Prompt)",
            8,
            22,
            12,
            key="prompt_size",
        )
        prompt_color = st.color_picker("Couleur (Prompt)", "#1E1E1E", key="prompt_color")
        prompt_bold = st.checkbox("Gras (Prompt)", value=True, key="prompt_bold")
        prompt_italic = st.checkbox("Italique (Prompt)", value=False, key="prompt_italic")

        st.subheader("Style de la Réponse")
        reponse_font = st.selectbox(
            "Police (Réponse)",
            ["Arial", "Calibri", "Times New Roman"],
            key="reponse_font",
        )
        reponse_size = st.slider(
            "Taille (Réponse)",
            8,
            22,
            12,
            key="reponse_size",
        )
        reponse_color = st.color_picker("Couleur (Réponse)", "#1E1E1E", key="reponse_color")
        reponse_bold = st.checkbox("Gras (Réponse)", value=False, key="reponse_bold")
        reponse_italic = st.checkbox("Italique (Réponse)", value=False, key="reponse_italic")


# =============================================================================
# Interface principale
# =============================================================================

# En-tête
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🤖 IA Multi-Modèles")
    st.caption("Interface unifiée - Version simplifiée")

# Tab principal
st.markdown("### 💬 Chat")

# Afficher l'historique en mode conversation
if st.session_state.conversation_mode and st.session_state.messages:
    st.subheader("Historique de conversation")
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
                st.caption(msg['timestamp'])
        else:
            with st.chat_message("assistant"):
                st.write(msg['content'])
                st.caption(msg['timestamp'])
    st.divider()

# Zone de saisie
st.subheader("Instruction et contexte")

user_instruction = st.text_area(
    "Votre instruction :",
    height=100,
    placeholder="Ex: Résume ce document en 5 points clés.",
)
uploaded_file = st.file_uploader(
    "Ajouter un document comme contexte (optionnel)", type=["docx", "pdf"]
)

prompt_final = user_instruction
prompt_text = user_instruction
document_structure_pour_export = None

if uploaded_file is not None:
    contenu_brut, template_styles = importer.analyser_document(uploaded_file)
    st.session_state.source_template_styles = template_styles

    # Gérer le cas du PDF qui retourne du texte brut
    if isinstance(contenu_brut, str):
        texte_a_traiter = contenu_brut
        st.warning("⚠️ L'analyse de style n'est pas supportée pour les PDF.")
        prompt_final = (
            "Voici une instruction à appliquer sur le contenu d'un document.\n\n"
            f'Instruction de l\'utilisateur : "{user_instruction}"\n\n'
            "Contenu du document à analyser :\n"
            f"{texte_a_traiter}\n\n"
            "---\nInstruction de formatage : Structure ta réponse finale en utilisant la syntaxe Markdown."
        )
    # Gérer le cas du DOCX qui retourne un dictionnaire structuré
    elif isinstance(contenu_brut, dict):
        document_structure_pour_export = contenu_brut
        json_str = json.dumps(contenu_brut, ensure_ascii=False, indent=2)

        prompt_final = (
            "INSTRUCTION : Tu es un assistant expert en traduction. Tu dois traduire le contenu textuel trouvé à l'intérieur d'une structure de données JSON.\n"
            f"TÂCHE DE TRADUCTION : \"{user_instruction}\"\n"
            "RÈGLE ABSOLUE : Tu dois impérativement conserver la structure JSON d'origine à l'identique (toutes les clés et leur hiérarchie : 'header', 'body', 'type', 'runs', 'style', etc.).\n"
            "RÈGLE ABSOLUE : Ne traduis UNIQUEMENT que les valeurs textuelles associées à la clé 'text'. Toutes les autres valeurs (comme 'font_name', 'is_bold', etc.) doivent rester inchangées.\n"
            "RÈGLE ABSOLUE : Ta réponse finale ne doit contenir QUE le JSON traduit, sans aucun texte, commentaire, ou explication avant ou après.\n\n"
            f"JSON À TRAITER :\n{json_str}"
        )
else:
    st.session_state.source_template_styles = None

prompt = prompt_final

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    generate_button = st.button("🚀 Générer", type="primary", use_container_width=True)

with col2:
    if st.session_state.conversation_mode:
        clear_button = st.button("🔄 Nouveau chat", use_container_width=True)
        if clear_button:
            clear_conversation()
            st.rerun()

# Génération de la réponse
if generate_button:
    if not user_instruction.strip():
        st.error("Veuillez saisir une instruction.")
    elif not api_key:
        st.error(f"⚠️ Veuillez entrer une clé API {provider}")
    else:
        try:
            # Obtenir le provider
            provider_instance = manager.get_provider(selected_model, api_key)

            # Préparer les paramètres selon le modèle
            if selected_model.startswith("gpt-5"):
                # Paramètres GPT-5
                params = {
                    'reasoning_effort': reasoning_effort,
                    'verbosity': verbosity,
                    'max_tokens': max_tokens
                }
                # Ajouter temperature seulement en mode minimal
                if reasoning_effort == "minimal" and 'temperature' in locals() and temperature is not None:
                    params['temperature'] = temperature
            else:
                # Paramètres classiques
                params = {
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }

                # Ajouter les paramètres avancés s'ils existent
                if 'top_p' in locals():
                    params['top_p'] = top_p
                if 'frequency_penalty' in locals():
                    params['frequency_penalty'] = frequency_penalty
                if 'presence_penalty' in locals():
                    params['presence_penalty'] = presence_penalty

            # Ajouter les paramètres spécifiques pour Claude
            if selected_model == "claude-sonnet-4-20250514" and 'use_thinking' in locals() and use_thinking:
                params['thinking_budget'] = thinking_budget

            if execution_mode == 'Réponse immédiate (Synchrone)':
                with st.spinner(f"Génération avec {selected_model}..."):
                    # Générer la réponse
                    if st.session_state.conversation_mode and st.session_state.messages:
                        # Mode conversation
                        add_message("user", prompt)
                        messages_for_api = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages
                        ]
                        response = provider_instance.chatter(messages_for_api, **params)
                        add_message("assistant", response)
                    else:
                        # Mode simple
                        response = provider_instance.generer_reponse(prompt, **params)

                # Incrémenter le compteur
                st.session_state.generation_count += 1

                # Afficher la réponse
                st.success("✅ Réponse générée avec succès!")

                styles_interface = {
                    "response": {
                        "font_name": reponse_font,
                        "font_size": reponse_size,
                        "font_color_rgb": hex_to_rgb(reponse_color),
                        "is_bold": reponse_bold,
                        "is_italic": reponse_italic,
                    }
                }

                with st.container():
                    st.markdown("### 🤖 Réponse")

                    if isinstance(document_structure_pour_export, dict):
                        try:
                            reponse_structuree = json.loads(response)
                            st.json(reponse_structuree)
                            buffer = exporter.generer_export_docx(
                                reponse_structuree, styles_interface
                            )
                        except json.JSONDecodeError:
                            st.error(
                                "L'IA n'a pas retourné une structure JSON valide. L'export utilisera une mise en forme basique."
                            )
                            st.write(response)
                            buffer = exporter.generer_export_docx_markdown(
                                response, styles_interface
                            )
                    else:
                        st.write(response)
                        buffer = exporter.generer_export_docx_markdown(
                            response, styles_interface
                        )

                    st.download_button(
                        "⬇️ Export DOCX",
                        data=buffer.getvalue(),
                        file_name="traduction.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                    # Métadonnées
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"Modèle: {selected_model}")
                    with col2:
                        if selected_model.startswith("gpt-5"):
                            st.caption(f"Reasoning: {reasoning_effort}")
                        else:
                            st.caption(f"Temperature: {temperature}")
                    with col3:
                        st.caption(f"Tokens max: {max_tokens}")

                # Option de copie
                st.code(response, language=None)
            else:
                try:
                    with st.spinner(f"Soumission du lot vers {selected_model}..."):
                        request_body = {
                            "model": selected_model,
                            "messages": [{"role": "user", "content": prompt}],
                            **params
                        }

                        batch_request = BatchRequest(
                            custom_id=f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            body=request_body,
                            prompt_text=prompt_text
                        )

                        batch_id = provider_instance.submit_batch(requests=[batch_request])

                    st.success("✅ Tâche soumise avec succès en traitement par lot.")
                    st.info(f"**ID du lot (Batch ID) :** `{batch_id}`")
                    st.caption("Vous pouvez suivre son état dans l'onglet 'Suivi des lots'.")
                except APIError as e:
                    st.error(f"❌ Erreur lors de la soumission du lot : {e}")
                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")

        except APIError as e:
            st.error(f"❌ Erreur API: {e}")

            # Proposer un fallback
            st.warning("💡 Voulez-vous essayer avec un autre modèle?")

            # Déterminer un modèle de fallback approprié
            if selected_model.startswith("gpt"):
                other_model = "claude-sonnet-4-20250514"
            else:
                other_model = "gpt-4.1"

            if st.button(f"Essayer {other_model}"):
                st.session_state.last_model = other_model
                st.rerun()

        except Exception as e:
            st.error(f"❌ Erreur inattendue: {e}")

# Section de suivi des lots
st.divider()
with st.expander("Suivi des lots (Batches)"):
    st.subheader("Historique des tâches de fond")

    provider_type_for_batch = get_model_provider_name(selected_model).lower()
    if provider_type_for_batch not in {"openai", "anthropic"}:
        provider_type_for_batch = "openai"
    api_key_for_batch = get_api_key(selected_model) or ""

    batch_manager = BatchJobManager(
        api_key=api_key_for_batch, provider_type=provider_type_for_batch
    )

    if "batch_history" not in st.session_state:
        st.session_state["batch_history"] = batch_manager.get_history(limit=20)

    if st.button("🔄 Rafraîchir l'historique complet (via API)"):
        try:
            with st.spinner("Récupération de l'historique..."):
                st.session_state["batch_history"] = batch_manager.get_history(limit=20)
        except Exception as e:
            st.error(f"Impossible de récupérer l'historique : {e}")

    if not api_key_for_batch:
        st.warning(
            "Veuillez fournir une clé API pour un suivi complet via API. Affichage de l'historique local uniquement."
        )

    history = st.session_state["batch_history"]

    if not history:
        st.info("Aucun lot trouvé.")
    else:
        for batch in history:
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**ID :** `{batch['id']}`")
                    st.caption(f"Créé le : {batch.get('created_at', 'N/A')}")
                    st.caption(f"Fournisseur : {batch.get('provider', 'N/A').capitalize()}")
                    if batch.get('request_counts'):
                        counts = batch['request_counts']
                        total = counts.get('total', 'N/A')
                        succeeded = counts.get('succeeded', 'N/A')
                        failed = counts.get('errored', counts.get('failed', 'N/A'))
                        st.caption(
                            f"Requêtes : {succeeded} succès / {failed} échecs sur {total} total"
                        )

                with col2:
                    status = batch.get('unified_status', 'unknown').upper()
                    st.metric("Statut", status)

                with col3:
                    state_key = f"details_{batch['id']}"
                    if state_key not in st.session_state:
                        st.session_state[state_key] = False

                    if st.button("Voir détails", key=f"details_btn_{batch['id']}"):
                        st.session_state[state_key] = not st.session_state[state_key]

                    if status == 'RUNNING':
                        if st.button("Annuler", key=f"cancel_{batch['id']}", type="secondary"):
                            if batch_manager.cancel_batch(batch['id']):
                                st.success(f"Demande d'annulation pour le lot {batch['id']} envoyée.")
                                st.rerun()
                            else:
                                st.error("Échec de l'annulation.")
                    elif status == 'COMPLETED':
                        results_export = batch_manager.get_results(batch['id'])
                        if results_export:
                            # On prend la réponse du premier (et unique) résultat du lot
                            response_text = results_export[0].clean_response

                            styles_interface = {
                                "response": {
                                    "font_name": reponse_font,
                                    "font_size": reponse_size,
                                    "font_color_rgb": hex_to_rgb(reponse_color),
                                    "is_bold": reponse_bold,
                                    "is_italic": reponse_italic,
                                }
                            }

                            # Nouvelle logique : Tenter de parser le JSON en priorité
                            try:
                                # Essayer de charger la réponse comme une structure JSON
                                reponse_structuree = json.loads(response_text)
                                # Si ça réussit, générer le DOCX structuré
                                buffer = exporter.generer_export_docx(
                                    reponse_structuree, styles_interface
                                )
                                st.success("Le document traduit et formaté est prêt.")
                            except (json.JSONDecodeError, TypeError):
                                # Si ce n'est pas un JSON valide, on se rabat sur le mode texte brut
                                st.warning("La réponse n'était pas une structure valide, export en mode texte.")
                                buffer = exporter.generer_export_docx_markdown(
                                    response_text, styles_interface
                                )

                            st.download_button(
                                "⬇️ Export DOCX",
                                data=buffer.getvalue(),
                                file_name=f"batch_{batch['id']}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key=f"download_{batch['id']}",
                            )

                if st.session_state[state_key]:
                    with st.spinner(f"Récupération des résultats pour {batch['id']}..."):
                        results = batch_manager.get_results(batch['id'])
                        if results:
                            for res in results:
                                if res.status == 'succeeded':
                                    with st.expander(f"✅ Succès : {res.custom_id}", expanded=True):
                                        if getattr(res, 'clean_response', None):
                                            st.markdown("**Réponse extraite :**")
                                            st.markdown(res.clean_response)
                                            with st.expander("Voir la réponse JSON brute"):
                                                st.json(res.response)
                                        else:
                                            st.json(res.response)
                                else:
                                    with st.expander(f"❌ Échec : {res.custom_id}", expanded=True):
                                        st.json(res.error)
                        else:
                            st.info("Aucun résultat disponible pour ce lot (il est peut-être encore en cours).")

# Footer
st.divider()
st.caption("Module IA Provider Unifié - Version simplifiée")
st.caption(f"Support pour: {', '.join(manager.get_available_models())}")
