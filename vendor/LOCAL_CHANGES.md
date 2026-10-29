# Adaptations locales du paquet Qwen

Source : révision 022e286b98fbec7e1e916cb940cdf532cd9f488e.

Le modèle du tokenizer 25 Hz est chargé à la demande dans core/__init__.py et
inference/qwen3_tts_tokenizer.py. Le tokenizer est identifié par sa configuration
avant son enregistrement. Le modèle 12 Hz utilisé ici conserve ses calculs et ses
poids ; il ne charge plus les dépendances de l’ancien tokenizer inutilisé.
Les véritables erreurs restent affichées.

Le paquet est conservé comme dépendance interne de Faster 0.6B. La dépendance
Gradio et le point d’entrée de démonstration sont retirés : le convertisseur
utilise uniquement l’inférence locale et ses références vocales.

## Faster-Qwen

Version 0.3.2, révision a70afc0f81f7f5f8801c3227968f1102f43f211c.
Cette révision est la dernière avant la migration à Transformers 5.

L’adaptateur optionnel GGML, ses arguments, ses exports et les points
d’entrée de démonstration sont retirés du paquet local. Seul le chemin
PyTorch/CUDA utilisé par Faster 0.6B est conservé. Les calculs et graphes
de ce chemin ne sont pas modifiés par ce nettoyage.

- `generate.py` ajoute le booléen `ended_with_eos` aux mesures de génération.
- `model.py` expose ces mesures dans `last_generation_timing` pour que le
  convertisseur refuse un passage coupé par la limite de jetons ou du cache.
- `model.py` utilise `dtype` et récupère la fréquence depuis le getter du tokenizer,
  évitant deux avertissements de compatibilité sans masquer d'erreurs.

Les calculs du modèle et du décodeur ne sont pas modifiés par ces adaptations.
L'attention eager suit les benchmarks du moteur, avec cache statique de 3072
jetons. La référence SIWIS pré-calculée est fournie en mode ICL complet.
