from flask import Flask, request, jsonify
import os
import requests
import tempfile
import replicate

# Configurer l'API Replicate avec votre clé API
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

app = Flask(__name__)

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

        # Ajouter l'image à l'historique si elle est présente
        if image_url:
            image_path = download_image(image_url)
            if image_path:
                # Exécution du modèle Replicate avec l'image et le prompt
                output = replicate_client.run(
                    "justmalhar/meta-llama-3.2-11b-vision:d48ad671cbc5f6e0c848f455ac2ca7280953fe1cf4039a010968f1cb19b0936f",
                    input={"image": image_url, "top_p": 0.95, "prompt": prompt, "temperature": 0.3}
                )
                history.append({
                    "role": "user",
                    "parts": [image_url, prompt],
                })
            else:
                return jsonify({'message': 'Failed to download image'}), 500
        else:
            history.append({
                "role": "user",
                "parts": [prompt],
            })

        # Ajouter la réponse du modèle à l'historique
        history.append({
            "role": "model",
            "parts": [output],
        })

        # Retourner la réponse du modèle avec un titre spécifique
        return jsonify({'message': f'☂️🐈 AI IMAGE 🐈✅ {output}'})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'message': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
