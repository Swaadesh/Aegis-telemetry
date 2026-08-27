"""
Backend for the Gen-Z Finance Brainrot Roaster.

Holds the Grok (xAI) API key server-side and generates the roast text.
The frontend (index.html) calls POST /roast with categorized transaction
data; this server builds a prompt, calls Grok, and returns the roast.

Why a backend at all: an API key can never be embedded in client-side
JS/HTML, since anyone can open dev tools and steal it. This server is
the only thing that ever sees the key.

Setup:
    1. pip install -r requirements.txt
    2. export XAI_API_KEY="your-key-here"     (get one at console.x.ai)
    3. python server.py
    4. Open index.html in your browser (it calls http://localhost:5000)
"""

import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow the HTML file (opened via file://) to call this server

XAI_API_KEY = os.environ.get("XAI_API_KEY")
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

SLANG_GLOSSARY = (
    "rizz, W, L, cooked, let him cook, bro is cooking, no cap, cap, fr, ngl, "
    "lowkey, highkey, based, valid, sus, mid, fire, goated, aura, aura points, "
    "yapping, yap session, delulu, touch grass, chronically online, NPC, "
    "main character, side quest, locked in, crash out, tweaking, wild, "
    "unhinged, it's giving..., vibe check, rent free"
)

SYSTEM_PROMPT = f"""You are the Roast Bot, a Gen-Z finance roast assistant.
You are given a user's REAL, categorized transaction data. Your job is to
roast their spending habits in a funny, savage, but ultimately supportive
Gen-Z voice using slang like: {SLANG_GLOSSARY}.

STRICT RULES:
- Only use the exact numbers and categories given to you. NEVER invent,
  estimate, or round dramatically. Every rupee amount you mention must
  come directly from the data provided.
- Never comment on categories not present in the data.
- Do not give real financial, medical, legal, or investment advice.
- Do not shame the user for things outside spending (appearance, etc).
- If a transaction looks like it could be a scam, fraud, or a large
  one-off loss rather than a habit, do NOT roast it. Flag it gently and
  supportively instead.
- Keep it to 5-7 short lines: one intro line, one line per top category,
  one closing verdict line.
- Output plain text lines only, no markdown, no numbering.
"""


@app.route("/roast", methods=["POST"])
def roast():
    if not XAI_API_KEY:
        return jsonify({
            "error": "Server is missing XAI_API_KEY. Set it as an environment variable before starting the server."
        }), 500

    data = request.get_json(force=True)
    transactions = data.get("transactions", [])

    if not transactions:
        return jsonify({"error": "No transactions provided."}), 400

    # Build a compact, factual summary the model must ground itself in
    category_totals = {}
    for t in transactions:
        cat = t.get("category", "Other")
        amt = t.get("amount", 0)
        category_totals[cat] = category_totals.get(cat, 0) + amt

    summary_lines = [
        f"{cat}: ₹{amt:,.0f}" for cat, amt in
        sorted(category_totals.items(), key=lambda x: -x[1])
    ]
    grand_total = sum(category_totals.values())

    user_prompt = (
        "Here is the user's real categorized spending data. Roast it using "
        "only these numbers, do not invent anything:\n\n"
        + "\n".join(summary_lines)
        + f"\n\nTotal spend: ₹{grand_total:,.0f}"
    )

    try:
        response = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
                "max_tokens": 400,
            },
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
        roast_text = result["choices"][0]["message"]["content"]
        return jsonify({"roast": roast_text, "grounded_totals": category_totals})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Grok API request failed: {str(e)}"}), 502
    except (KeyError, IndexError) as e:
        return jsonify({"error": f"Unexpected response format from Grok: {str(e)}"}), 502


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "api_key_configured": bool(XAI_API_KEY)})


if __name__ == "__main__":
    if not XAI_API_KEY:
        print("WARNING: XAI_API_KEY is not set. /roast will return an error until you set it.")
    app.run(host="0.0.0.0", port=5000, debug=True)
