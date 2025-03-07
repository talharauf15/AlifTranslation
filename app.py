from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import gradio as gr

# Load the model and tokenizer
model_path = r"arabic_to_english_model"
tokenizer_path = r"arabic_to_english_tokenizer"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define the translation function
def translate_arabic_to_english(arabic_text):
    # Tokenize the input text
    inputs = tokenizer(arabic_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device
    
    # Generate translation
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text

# Custom CSS for styling
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    padding: 20px;
    border-radius: 10px;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
"""

# Create a Gradio Blocks interface
with gr.Blocks(css=custom_css) as demo:
    # Add a title and description
    gr.Markdown("# AlifTrans 🌍📖")
    gr.Markdown("Translate Arabic text to English using a state-of-the-art Transformer model. Simply type or paste your Arabic text below and click 'Translate'!")

    # Input and output components
    with gr.Row():
        arabic_input = gr.Textbox(lines=3, placeholder="Enter Arabic text here...", label="Arabic Text")
        english_output = gr.Textbox(lines=3, label="Translated English Text")

    # Translate button
    translate_button = gr.Button("Translate 🚀")

    # Examples for users to try
    gr.Examples(
        examples=[
            ["مرحبًا كيف حالك؟", "Hello, how are you?"],
            ["أنا سعيد لأنني أتعلم الذكاء الاصطناعي.", "I am happy because I am learning artificial intelligence."],
            ["اللغة العربية جميلة.", "The Arabic language is beautiful."],
        ],
        inputs=arabic_input,
        outputs=english_output,
        fn=translate_arabic_to_english,
        cache_examples=True,
        label="Try these examples!",
    )

    # Link the button to the translation function
    translate_button.click(fn=translate_arabic_to_english, inputs=arabic_input, outputs=english_output)

    # Footer
    gr.Markdown("---")
    gr.Markdown("### Made with ❤️ using [Gradio](https://gradio.app) and [Hugging Face Transformers](https://huggingface.co/transformers/)")


# Launch the Gradio app
demo.launch()