# pip install torch transformers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

instruction = "given a dialog context, you need to respond empathically"
knowledge = input("Provide related knowledge (or leave empty if none): ")

def predict(input_text, history=[]):
    s = list(sum(history, ()))
    s.append(input_text)
    dialog = ' EOS '.join(s)

    if knowledge == "":
        query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    else:
        query = f"{instruction} [CONTEXT] {dialog} [KNOWLEDGE] {knowledge}"

    top_p = 0.9
    min_length = 8
    max_length = 64

    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(f"{query}", return_tensors='pt')

    output = model.generate(new_user_input_ids, min_length=int(min_length), max_length=int(max_length), top_p=top_p, do_sample=True).tolist()
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    history.append((input_text, response))

    return response, history

def main():
    history = []
    while True:
        input_text = input("You: ")
        if input_text.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        response, history = predict(input_text, history)
        print("GODEL:", response)

if __name__ == "__main__":
    main()
