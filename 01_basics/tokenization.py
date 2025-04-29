# Tokenization
import tiktoken

# Vocab Size
encoder = tiktoken.encoding_for_model("gpt-4o")
print("Vocab Size: ", encoder.n_vocab) #Vocab Size:  200019

# Encode
text = "The cat sat on the mat."
print("Text: ", text)
tokens = encoder.encode(text)
print("Encode: ", tokens)

# Decode
decoder = encoder.decode(tokens)
print("Decode: ", decoder)
