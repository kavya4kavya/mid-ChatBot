from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Using Google's FLAN. Can modify this to use any other model too
model_Name="google/flan-t5-base"

model=AutoModelForSeq2SeqLM.from_pretrained(model_Name)
tokenizer=AutoTokenizer.from_pretrained(model_Name)

conversation_history=[]

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    
