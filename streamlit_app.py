import os
from dotenv import load_dotenv
import streamlit as st
import requests
import json
import chromadb
from openai import OpenAI
import random
from pathlib import Path
import base64

# --- CONFIGURATION & API KEYS ---

load_dotenv(".env")

# Retrieve API keys from environment variables

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EDAMAM_APP_ID = os.environ.get('EDAMAM_APP_ID')
EDAMAM_APP_KEY = os.environ.get('EDAMAM_APP_KEY')
EDAMAM_ACCOUNT_USER = os.environ.get('EDAMAM_ACCOUNT_USER')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- USER-FACING ERROR MESSAGE ---
APOLOGY_MESSAGE = "Sorry, I am not able to help you with that. Please click on 'Start new chat' to try again."

# --- CHROMADB RAG SETUP ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="heart_guidelines")
# The PDF loading logic needs to run in the main Colab notebook, not inside the Streamlit app
# For the Streamlit app, we assume the collection is already populated.
# If the collection is empty, load a sample to ensure functionality.
if collection.count() == 0:
    sample_text = (
        "According to the NHLBI Keep the Beat guidelines, adopting a heart-healthy eating plan "
        "involves choosing foods low in saturated fats, trans fats, cholesterol, and sodium. "
        "Substitute solid fats like butter, stick margarine, and shortening with liquid vegetable oils "
        "such as olive, canola, or safflower oil. Focus on whole grains, fiber, and lean proteins."
    )
    collection.add(documents=[sample_text], metadatas=[{"source": "NHLBI Guidelines"}], ids=["doc_1"])

# --- SPOKE TOOLS (PYTHON FUNCTIONS) ---
def search_heart_healthy_recipes(query_keywords, specific_ingredients=None, num_recipes_to_return=5):
    """Searches the Edamam API for heart-healthy recipes (low fat/low sodium) and returns a random sample of recipes.

    Args:
        query_keywords (str): The main keywords for the recipe search.
        specific_ingredients (list, optional): A list of specific ingredients to include. Defaults to None.
        num_recipes_to_return (int, optional): The number of random recipes to return from the API results. Defaults to 5.

    Returns:
        str: A JSON string of a random sample of heart-healthy recipes.
    """
    url = "https://api.edamam.com/api/recipes/v2"
    params = {
        "type": "public",
        "q": query_keywords,
        "app_id": EDAMAM_APP_ID,
        "app_key": EDAMAM_APP_KEY,
        "diet": ["low-fat", "low-sodium"], # Hardcoded dietary restrictions
        "dishType": [
            "Biscuits and cookies", "Bread", "Cereals",
            "Main course", "Pancake", "Pasta", "Pastry", "Pies and tarts",
            "Pizza", "Salad", "Sandwiches", "Side dish", "Soup", "Starter", "Egg", "Preserve"
        ]
    }

    headers = {
        "Edamam-Account-User": EDAMAM_ACCOUNT_USER
    }

    if specific_ingredients:
        params['ingredient'] = specific_ingredients

    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_recipes = []
            for hit in data.get("hits", []):
                recipe = hit["recipe"]
                all_recipes.append({
                    "name": recipe["label"],
                    "calories": round(recipe["calories"]),
                    "url": recipe["url"],
                    "ingredients": recipe["ingredientLines"]
                })
            
            if len(all_recipes) > num_recipes_to_return:
                filtered_recipes = random.sample(all_recipes, num_recipes_to_return)
            else:
                filtered_recipes = all_recipes

            return json.dumps(filtered_recipes)
        else:
            return json.dumps({"error": f"Edamam API error: {response.status_code} - {response.text}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_plating_image(recipe_name):
    enhanced_prompt = f"Gourmet food photography of {recipe_name}, full plated meal entirely visible, the whole plate in frame, centered, clean overhead or slightly angled view, modern plain white dish, healthy heart meal, cinematic lighting."
    try:
        response = client.images.generate(
            model="gpt-image-2",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )

        if not response or not getattr(response, "data", None) or len(response.data) == 0:
            return {"type": "error", "value": "Image generation returned an empty response."}

        img = response.data[0]

        # Case 1: URL returned
        if getattr(img, "url", None):
            return {"type": "url", "value": img.url}

        # Case 2: base64 returned
        if getattr(img, "b64_json", None):
            image_bytes = base64.b64decode(img.b64_json)
            return {"type": "bytes", "value": image_bytes}

        return {"type": "error", "value": "Image was generated but no displayable payload was returned."}
    except Exception as e:
        return {"type": "error", "value": f"Error generating image: {str(e)}"}

def generate_conversational_response(question, context_chunks):
    """Generates a conversational response using an LLM based on the question and retrieved context."""
    if not context_chunks or context_chunks == "No specific guidelines found for this topic.":
        return "I couldn't find specific guidelines related to your question in the provided documents."

    prompt = f"""You are a helpful assistant providing information about dietary guidelines.
Based on the following context, answer the user's question in a conversational manner.
If the answer is not available in the context, politely state that you cannot provide an answer from the given information.

Context:
{context_chunks}

User Question: {question}
Conversational Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing information about dietary guidelines."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating conversational response: {e}")
        return "I apologize, but I encountered an error while trying to generate a conversational response."

def get_dietary_guidelines(question, n_results=3):
    """Queries ChromaDB for relevant heart-healthy guidelines and generates a conversational answer."""
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved_chunks = results['documents'][0] if results['documents'] else []

    if not retrieved_chunks:
        context_text = "No specific guidelines found for this topic."
    else:
        context_text = "\n---\n".join(retrieved_chunks)

    return generate_conversational_response(question, context_text)

# --- OPENAI NATIVE TOOL SCHEMAS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_heart_healthy_recipes",
            "description": "Use this to find specific low-fat, low-sodium, heart-healthy recipes based on user ingredients or cravings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_keywords": {
                        "type": "string",
                        "description": "Main keywords for the recipe search (e.g., 'salmon', 'quinoa salad')."
                    },
                    "specific_ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific ingredients to include."
                    },
                    "num_recipes_to_return": {
                        "type": "integer",
                        "description": "Number of recipes to return (default 5)."
                    }
                },
                "required": ["query_keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_plating_image",
            "description": "Use this to generate a visual image of the final plated meal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "The name of the recipe to visualize."
                    }
                },
                "required": ["recipe_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dietary_guidelines",
            "description": "Use this to answer general medical, nutritional, or guideline questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about dietary guidelines or heart health."
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of guideline chunks to retrieve (default 3)."
                    }
                },
                "required": ["question"]
            }
        }
    }
]

# --- STREAMLIT UI & LIMITED MEMORY ---

base_dir = Path(__file__).resolve().parent
logo_path = base_dir / "HeartyEatsAppLogo.png"

# App logo

if logo_path.exists():
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(str(logo_path), width=360)
else:
    st.warning(f"Logo not found at {logo_path}")

# Start new chat button

with st.sidebar:
    st.markdown("### Session")
    if st.button("🆕 Start a new chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "system", "content": "You are a heart-healthy culinary assistant. Help users find delicious recipes, check guidelines, and visualize plating concepts."}
        ]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a heart-healthy culinary assistant. Help users find delicious recipes, check guidelines, and visualize plating concepts."}
    ]

for msg in st.session_state.messages[1:]:
    if isinstance(msg, dict) and msg.get("role") == "tool":
        continue
    if isinstance(msg, dict) and "content" in msg and msg["content"]:
        st.chat_message(msg["role"]).write(msg["content"])
    elif hasattr(msg, "content") and msg.content:
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input("Ask for a recipe, dietary guideline, or plating visual..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Show loading message while processing
    with st.spinner("⏳ I'm working on it, please wait..."):
        memory_window = [st.session_state.messages[0]] + st.session_state.messages[-5:]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=memory_window,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"Initial API call error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": APOLOGY_MESSAGE})
            st.chat_message("assistant").write(APOLOGY_MESSAGE)
            st.stop()

        response_message = response.choices[0].message

        if response_message.tool_calls:
            st.session_state.messages.append(response_message)
            had_error = False

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments or "{}")
                tool_result = ""

                try:
                    # Execute the tool and get result
                    if function_name == "search_heart_healthy_recipes":
                        tool_result = search_heart_healthy_recipes(
                            function_args.get("query_keywords"),
                            function_args.get("specific_ingredients"),
                            function_args.get("num_recipes_to_return", 5),
                        )

                    elif function_name == "generate_plating_image":
                        image_result = generate_plating_image(function_args.get("recipe_name"))

                        if isinstance(image_result, dict):
                            if image_result["type"] == "url":
                                st.image(image_result["value"], caption=f"Plating Concept: {function_args.get('recipe_name')}")
                                tool_result = image_result["value"]
                            elif image_result["type"] == "bytes":
                                st.image(image_result["value"], caption=f"Plating Concept: {function_args.get('recipe_name')}")
                                tool_result = "Image generated and displayed from base64 payload."
                            else:
                                print(f"Image generation error: {image_result['value']}")
                                had_error = True
                                tool_result = APOLOGY_MESSAGE
                        else:
                            print("Unexpected image response format.")
                            had_error = True
                            tool_result = APOLOGY_MESSAGE

                    elif function_name == "get_dietary_guidelines":
                        tool_result = get_dietary_guidelines(
                            function_args.get("question"),
                            function_args.get("n_results", 3),
                        )

                    else:
                        print(f"Unknown tool called: {function_name}")
                        had_error = True
                        tool_result = APOLOGY_MESSAGE

                except Exception as e:
                    print(f"Tool execution error in {function_name}: {e}")
                    had_error = True
                    tool_result = APOLOGY_MESSAGE

                finally:
                    # Add tool result message to conversation history for OpenAI
                    # CRITICAL: always respond to every tool_call_id
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                    })

            if had_error:
                st.session_state.messages.append({"role": "assistant", "content": APOLOGY_MESSAGE})
                st.chat_message("assistant").write(APOLOGY_MESSAGE)
            else:
                # After processing all tool calls, make a follow-up request for the final response
                memory_window = [st.session_state.messages[0]] + st.session_state.messages[-10:]

                try:
                    final_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=memory_window,
                        tools=tools,
                        tool_choice="auto"
                    )
                except Exception as e:
                    print(f"Final API call error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": APOLOGY_MESSAGE})
                    st.chat_message("assistant").write(APOLOGY_MESSAGE)
                    st.stop()

                final_message = final_response.choices[0].message
                content = final_message.content or ""
                st.session_state.messages.append({"role": "assistant", "content": content})
                if content:
                    st.chat_message("assistant").write(content)
        else:
            content = response_message.content or ""
            st.session_state.messages.append({"role": "assistant", "content": content})
            st.chat_message("assistant").write(content)
