import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load reviews dataset
df = pd.read_csv("reviews.csv")

# Few-Shot Chain-of-Thought Prompt
prompt_template = """
You are a sentiment analysis assistant.

Follow these steps carefully:

1. Identify all positive-sentiment phrases.
2. Identify all negative-sentiment phrases.
3. Check for contradictions or mixed sentiment.
4. Decide the final sentiment label:
   - positive
   - neutral
   - negative
5. Explain the reasoning step by step.

Example 1:

Review:
"The phone camera is excellent and battery lasts all day."

Positive phrases:
- "camera is excellent"
- "battery lasts all day"

Negative phrases:
- None

Mixed sentiment:
- No contradictions found

Final reasoning:
- The review contains strong positive opinions about the phone features.

Sentiment:
- positive


Example 2:

Review:
"The product is okay but delivery was delayed."

Positive phrases:
- "product is okay"

Negative phrases:
- "delivery was delayed"

Mixed sentiment:
- Both positive and negative aspects exist.

Final reasoning:
- The review has balanced opinions with both satisfaction and dissatisfaction.

Sentiment:
- neutral


Example 3:

Review:
"Terrible quality and customer service never helped."

Positive phrases:
- None

Negative phrases:
- "Terrible quality"
- "customer service never helped"

Mixed sentiment:
- No positive indicators found.

Final reasoning:
- The review expresses strong dissatisfaction with the product and service.

Sentiment:
- negative


Now analyze this review.

Return the result EXACTLY in this format:

Positive phrases:
- ...

Negative phrases:
- ...

Mixed sentiment:
- ...

Final reasoning:
- ...

Sentiment:
- positive / neutral / negative


Review:
"{review}"
"""

# Analyze each review
for review in df["review"]:

    prompt = prompt_template.format(review=review)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    print("\n===================================")
    print("REVIEW:")
    print(review)

    print("\nMODEL OUTPUT:")
    print(response.choices[0].message.content)
