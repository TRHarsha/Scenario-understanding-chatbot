import streamlit as st
from transformers import FlaxT5ForConditionalGeneration, T5Tokenizer
import jax.numpy as jnp

# Load the FLAN-T5 model and tokenizer using FLAX (JAX backend)
model_name = "google/flan-t5-large"
model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Preloaded main query
main_query = "Ask me to explain the scenario, based on the scenario ask questions as plaintiff should have each and every little detail of the case. Ask questions till you are satisfied to prepare the plaintiff notice. Once you have the answers needed, prepare a plaintiff statement."

# Initialize session state for tracking conversation
if "questions" not in st.session_state:
    st.session_state.questions = ["Please explain the scenario in detail:"]
    st.session_state.responses = []
    st.session_state.complete = False

st.title("Dynamic Plaintiff Notice Preparation")

# Function to generate the next question based on user-provided context
def generate_next_question(context):
    prompt = "Generate a follow-up question based on the context: " + context
    inputs = tokenizer(prompt, return_tensors="jax", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    next_question = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return next_question

# Display the current question and collect user's answer
if not st.session_state.complete:
    # Show the last question in session state
    current_question = st.session_state.questions[-1]
    user_answer = st.text_input(current_question, key=current_question)

    # If the user provides an answer, generate the next question or finalize
    if user_answer:
        # Store the answer and prepare context for the next question
        st.session_state.responses.append(user_answer)
        context = " ".join([f"{q}: {a}" for q, a in zip(st.session_state.questions, st.session_state.responses)])

        # Generate the next question
        next_question = generate_next_question(context)
        
        # Check for stopping criteria (modify as needed)
        if "I'm satisfied with the information" in next_question or len(st.session_state.questions) > 10:
            st.session_state.complete = True
        else:
            st.session_state.questions.append(next_question)

# Once the information gathering is complete, prepare the plaintiff notice
if st.session_state.complete:
    st.subheader("Plaintiff Notice:")
    plaintiff_statement = f"PLAINTIFF NOTICE\n\nRegarding the scenario: {main_query}\n\n"
    for question, answer in zip(st.session_state.questions, st.session_state.responses):
        plaintiff_statement += f"{question}\n{answer}\n\n"
    plaintiff_statement += "This notice has been prepared based on the information provided above."

    st.write(plaintiff_statement)
else:
    st.write("Please provide responses to prepare a detailed plaintiff notice.")
