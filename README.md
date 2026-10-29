# Audiobook Maker

Conversion locale d’EPUB en livres audio avec **Faster-Qwen3-TTS 0.3.2** et
**Qwen3-TTS 0.6B Base**, sur carte NVIDIA. Ce sont le seul moteur de synthèse
et le seul modèle utilisés par le programme.

## Installation et lancement

Prérequis : **Windows 10/11 x64**, **Python 3.12 x64** et une carte NVIDIA
compatible bfloat16 et PyTorch CUDA 12.4, avec son pilote installé. La
configuration est validée sur RTX 4060. Python est disponible sur
[python.org](https://www.python.org/downloads/windows/). Le CUDA Toolkit et
Visual Studio ne sont pas nécessaires. Prévoir une connexion Internet et
plusieurs Go disponibles pour la première installation.

1. Télécharger ou cloner le dépôt dans un dossier accessible en écriture.
2. Double-cliquer sur **`installer.bat`** et attendre la confirmation finale.
3. Double-cliquer sur **`start_ebook.bat`**. Le livre de démonstration est déjà
   présent dans `input/test` ; ajouter ses propres livres dans `input`.

L’installateur réalise automatiquement ces opérations, sans droits
administrateur et sans modifier les autres environnements Python :

- création de `.venv` dans le projet ;
- installation de PyTorch et torchaudio **2.6.0 avec CUDA 12.4**, puis des
  bibliothèques locales et des dépendances aux versions fixées ;
- téléchargement des **11 fichiers** de **Qwen3-TTS 0.6B Base**, tokenizer
  audio compris, à la révision exacte de `model_sources.json` ;
- vérification de la taille et du **SHA-256 de chaque fichier** ;
- installation locale de **FFmpeg et FFprobe**, avec archive vérifiée ;
- création de `input`, `input/test` et `output` ;
- copie du premier chapitre complet de *Vingt mille lieues sous les mers*,
  « Un écueil fuyant », depuis le petit EPUB fourni dans `assets/samples` ;
- vérification des dépendances, de CUDA, de la voix et des codecs de sortie.

Aucun compte Hugging Face, Git LFS ou téléchargement du modèle 1.7B n’est
nécessaire. Les poids occupent environ **2,52 Go**, dans
`models/Qwen3-TTS-0.6B-Base`. Les sources, versions et empreintes sont fixées
dans `model_sources.json` et `installation_sources.json`.

Une nouvelle exécution réutilise les poids déjà vérifiés. Elle ne remplace
pas un EPUB de même nom dans `input/test` et ne touche pas aux livres ou aux
audios personnels. Après une interruption, relancer `installer.bat`.
Les détails de pip sont conservés dans `.cache/pip-install.log` et le résultat
des contrôles dans `.cache/installation.json`. Aucun livre n’est synthétisé
pendant l’installation.

```powershell
.\installer.bat --check
.\installer.bat --no-pause
```

`--check` vérifie l’installation existante sans téléchargement ni génération
TTS. `--no-pause` permet une exécution automatisée. Si le lanceur Python n’est
pas disponible, installer Python 3.12 x64, ou définir `AUDIOBOOK_PYTHON` avec le
chemin de son interpréteur avant de lancer `installer.bat`.

Les dossiers s’affichent avant les EPUB. Un numéro ouvre un dossier ou choisit
un livre. Plusieurs EPUB se sélectionnent avec `1 3 5`, `1,3,5` ou `1-3 5`.
`0` ou `..` remonte au dossier parent ; Entrée choisit tous les livres du
dossier courant, sous-dossiers compris. `output` reproduit cette arborescence.

Faster 0.6B est utilisé directement, sans menu de moteur ou de taille de
modèle. Les questions portent sur la voix, le batching, le découpage des
segments, les chapitres et la sortie audio.

```powershell
.\start_ebook.bat "input\roman.epub" --batch-size 6 --yes
.\start_ebook.bat "input\romans" --batch-size 7 --yes
.\start_ebook.bat "input\roman.epub" --no-batching --format wav --yes
.\start_ebook.bat --list-voices
```

Les anciens arguments `--engine` et `--model-size` sont retirés. La génération
TTS utilise les poids locaux. `--offline` interdit les téléchargements des
composants qui respectent ce réglage.

## Voix et génération

Seul l’échantillon SIWIS nº 4 est fourni dans `voices/references_fr` :
`expr_chap_s01_0004-7e5a77eb`, utilisé par défaut. Une référence associe un WAV
ou FLAC à une transcription `.txt` de même nom. Elle est préparée une seule
fois et réutilisée pour tous les passages. Des références personnelles peuvent
être ajoutées dans ce dossier ; elles restent exclues de Git.

Faster conserve **bfloat16**, l’attention **eager**, quatre threads PyTorch et
ses graphes CUDA. Le moteur reste chargé entre les chapitres et les livres.
La bibliothèque `vendor/Qwen3-TTS` reste nécessaire : Faster utilise ses
classes de modèle et de tokenizer. Elle ne constitue pas un choix de moteur
supplémentaire dans le convertisseur.

Le batching propose **2, 3, 4, 5, 6, 7 ou 8 passages**, ou un passage à la fois.
La valeur initiale est **6** et le dernier choix est mémorisé. Les lots sont
regroupés par longueur dans une fenêtre bornée, avec restitution des audios
dans l’ordre du livre.

Les optimisations sont conservées : cache GPU adaptatif **768 / 1024 / 1536 /
3072 positions** partageant la même allocation, masques et positions capturés,
embeddings du codec calculés dans un graphe, historique des jetons préalloué
et vérifications de fin groupées. Voir `vendor/LOCAL_CHANGES.md` et
`epub_audio/_vendor/faster_qwen3_tts_batch/NOTICE.md` pour les adaptations locales.

Un passage doit atteindre son jeton de fin. Les passages incomplets sont
repris individuellement puis divisés si nécessaire, sans perdre les mots.
Les limites restent définies par `qwen.json` et
`epub_audio/core/generation_budget.py`.

Le batching continu reste **expérimental et désactivé par défaut** :

```powershell
.\start_ebook.bat "input\roman.epub" --batch-size 6 --continuous-batching --yes
```

## Découpage et lecture

Le lecteur exploite la structure et la table des matières de l’EPUB, les
titres, les notes et les ruptures de scène. `--segmentation paragraph`
conserve les paragraphes ; `--segmentation sentence` découpe par phrases.
Les paragraphes longs sont répartis en passages plus courts.

L’introduction utilise les métadonnées du livre. Les titres sont préparés
pour être prononcés. `--convert-numbers` et `--no-convert-numbers` contrôlent
la conversion des nombres en lettres. Les pauses habituelles restent de
800 ms entre paragraphes, 1 500 ms aux ruptures de scène et 2 000 ms après
l’introduction. La voix garde son débit naturel ; `--rate 1.1` applique un
tempo FFmpeg après génération.

## Chapitres avec un LLM local

L’analyse optionnelle des chapitres utilise Ollama localement, avec
`qwen3.5:4b` par défaut. Elle sert au découpage du livre et ne remplace pas
le moteur TTS. Les seuls poids de synthèse sont ceux du modèle 0.6B.

```powershell
.\start_ebook.bat "input\roman.epub" --llm-chapters --yes
.\start_ebook.bat "input\roman.epub" --no-llm-chapters --yes
.\start_ebook.bat "input\roman.epub" --llm-chapters --chapters-only --yes
```

`--llm-model` et `--llm-url` configurent le modèle et le serveur Ollama.
L’analyse précède le chargement TTS. En cas d’échec, le découpage habituel est
conservé avec un diagnostic. `--chapters-only` exporte les plans JSON sans audio.

## Sorties, préférences et suivi

Formats disponibles : **WAV**, **MP3**, **AAC** et **Opus**. Le WAV PCM original
est mono 24 kHz, sans réencodage. FFmpeg et FFprobe sont installés dans
`.cache/ffmpeg/bin` et utilisés automatiquement, même sans modification du
PATH système. `--frequency` et `--bitrate` règlent la fréquence et le
débit. Opus accepte 24 ou 48 kHz en entrée et se décode à 48 kHz.

Un **master WAV original par chapitre** est conservé sous `output/master`,
dans l’arborescence correspondante. La normalisation optionnelle
(`--normalize`) vise −18 LUFS et −3 dBTP sur le chapitre complet. Elle est
désactivée par défaut et préserve les masters originaux.

La synthèse utilise un seul moteur GPU. Après la génération du livre,
les chapitres sont exportés en parallèle. `--workers` règle uniquement cet
export, entre 1 et 20, avec 20 par défaut. Le temps **génération TTS** est
affiché séparément du temps total comprenant chargement et export.

Les choix sont mémorisés dans `.cache/preferences.json`. La section `prompts`
peut masquer individuellement les questions `voice`, `batching`,
`segmentation`, `llm_chapters`, `convert_numbers`, `audio_format`, `bitrate`,
`sample_rate`, `normalize` et `confirmation`. Les anciennes préférences
migrent vers la dernière voix Faster en conservant le batching et les exports.

Journaux locaux :

- `.cache/qwen-last-run.json` : modèle, référence et configuration ;
- `.cache/qwen-segments.jsonl` : durée et temps de génération des passages ;
- `.cache/qwen-batch-failures.jsonl` : passages incomplets et erreurs des lots.

Les livres, références vocales et audios de `output` sont conservés. Les
poids, caches, données personnelles et médias générés restent exclus de Git.
Seuls l’EPUB public de démonstration et la référence SIWIS nº 4 sont
distribués. Leur provenance figure dans les README de
`assets/samples` et `voices/references_fr`.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Les tests utilisent des modèles simulés et de courts fichiers audio pour
vérifier le batching, les reprises, l’ordre des segments, les pauses,
les chapitres, les préférences, les masters et les exports. Ils ne lancent
pas de génération TTS d’un livre complet.
