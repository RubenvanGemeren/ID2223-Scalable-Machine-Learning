import gradio as gr

# Load model directly
from transformers import AutoModel, AutoTokenizer

# Load the LoRA model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ID2223JR/lora_model")
model = AutoModel.from_pretrained("ID2223JR/lora_model")

# Data storage
ingredients_list = []


# Function to add ingredient
def add_ingredient(ingredient, quantity):
    if ingredient and quantity > 0:
        ingredients_list.append(f"{ingredient}, {quantity} grams")
    return (
        "\n".join(ingredients_list),
        gr.update(value="", interactive=True),
        gr.update(value=None, interactive=True),
    )


# Function to enable/disable add button
def validate_inputs(ingredient, quantity):
    if ingredient and quantity > 0:
        return gr.update(interactive=True)
    return gr.update(interactive=False)


# Function to handle model submission
def submit_to_model():
    if not ingredients_list:
        return "Ingredients list is empty! Please add ingredients first."

    # Join ingredients into a single prompt
    prompt = f"Using the following ingredients, suggest a recipe:\n\n" + "\n".join(
        ingredients_list
    )

    # Tokenize and pass the prompt to the model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode the model output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# App
def app():
    with gr.Blocks() as demo:
        with gr.Row():
            ingredient_input = gr.Textbox(
                label="Ingredient", placeholder="Enter ingredient name"
            )
            quantity_input = gr.Number(label="Quantity (grams)", value=None)

        add_button = gr.Button("Add Ingredient", interactive=False)
        output = gr.Textbox(label="Ingredients List", lines=10, interactive=False)

        with gr.Row():
            submit_button = gr.Button("Submit")
            model_output = gr.Textbox(
                label="Recipe Suggestion", lines=10, interactive=False
            )

        # Validate inputs
        ingredient_input.change(
            validate_inputs, [ingredient_input, quantity_input], add_button
        )
        quantity_input.change(
            validate_inputs, [ingredient_input, quantity_input], add_button
        )

        # Add ingredient logic
        add_button.click(
            add_ingredient,
            [ingredient_input, quantity_input],
            [output, ingredient_input, quantity_input],
        )

        # Submit to model logic
        submit_button.click(
            submit_to_model,
            inputs=None,  # No inputs required as it uses the global ingredients_list
            outputs=model_output,
        )

    return demo


demo = app()
demo.launch()
