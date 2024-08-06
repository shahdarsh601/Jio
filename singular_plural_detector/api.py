from flask import Flask, request, jsonify
from singular_plural_detector import SingularPluralDetector

app = Flask(__name__)
detector = SingularPluralDetector()

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'sentence' not in data:
        return jsonify({"error": "No sentence provided"}), 400

    sentence = data['sentence']
    singulars, plurals = detector.detect_singular_plural(sentence)

    return jsonify({"singulars": singulars, "plurals": plurals})

if __name__ == "__main__":
    app.run(debug=True)
