from flask import Flask, request, jsonify
import os
import requests
import tempfile
import replicate

app = Flask(__name__)

# Initialiser l'API Replicate avec votre clé API
replicate_client = replicate.Client(api_token=os.environ.get('REPLICATE_API_TOKEN'))

# Dictionnaire pour stocker les historiques de conversation
sessions = {}

def download_image(url):
    """Télécharge une image depuis une URL et retourne le chemin du fichier temporaire."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            return temp_file.name
    else:
        return None

@app.route('/api/replicate', methods=['POST'])
def handle_request():
    try:
        data = request.json
        prompt = data.get('prompt', '')  # Question ou prompt de l'utilisateur
        custom_id = data.get('customId', '')  # Identifiant de l'utilisateur ou session
        image_url = data.get('link', '')  # URL de l'image

        # Récupérer l'historique de la session existante ou en créer une nouvelle
        if custom_id not in sessions:
            sessions[custom_id] = []  # Nouvelle session
        history = sessions[custom_id]

        # Télécharger l'image depuis l'URL fournie
        if image_url:
            image_path = download_image(image_url)
            if image_path:
                # Traiter l'image via Replicate avec le modèle spécifié
                output = replicate_client.run(
                    "justmalhar/meta-llama-3.2-11b-vision:d48ad671cbc5f6e0c848f455ac2ca7280953fe1cf4039a010968f1cb19b0936f",
                    input={
                        "image": image_url,
                        "top_p": 0.95,
                        "prompt": prompt,
                        "temperature": 0.3
                    }
                )

                # Ajouter l'entrée de l'utilisateur à l'historique
                history.append({
                    "role": "user",
                    "parts": [prompt, image_url],
                })

                # Ajouter la réponse du modèle à l'historique
                history.append({
                    "role": "model",
                    "parts": [output],
                })

                # Ajouter le titre à la réponse
                titled_response = f"☂️🐈 AI IMAGE 🐈✅ {output}"

                # Retourner la réponse du modèle avec le titre
                return jsonify({'message': titled_response})

            else:
                return jsonify({'message': 'Failed to download image'}), 500

        else:
            return jsonify({'message': 'No image URL provided'}), 400

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'message': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
