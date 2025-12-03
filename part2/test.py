from tokenizer import ByteTokenizer

tok  = ByteTokenizer()
s = "Xin chào các bạn!"
ids = tok.encode(s) 
print("Encoded IDs:", ids)
s_decoded = tok.decode(ids)
print("Decoded String:", s_decoded)
print("Vocab Size:", tok.vocab_size)



