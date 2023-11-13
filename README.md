# Genai chatbot example with GODEL-v1_1

This project implements a conversational AI using the `GODEL-v1_1-large-seq2seq` model from Microsoft, built on the Hugging Face `transformers` library. I recommend you look at other models available and choose one that you think would be good for whatever use case you have.

## Installation

Before running the script, ensure you have the necessary libraries:
```
pip install torch transformers
```

## What is Tokenization?

Tokenization is the process of converting text into a series of tokens, which are smaller pieces, like words or subwords. This is crucial in natural language processing as it helps in breaking down complex text into manageable parts. These tokens are then used by models like `GODEL-v1_1` to understand and generate human-like text. Proper tokenization ensures the model interprets and responds to inputs accurately.

## Usage

You can provide additional knowledge at the beginning to steer the context of the conversation. If you are using this code for the competition, make sure to give it context to your character through information such as character dialogue and interactions (as shown in the demo).
