from transformers import AutoTokenizer, T5EncoderModel

name = "google/byt5-small"
text_tokenizer = AutoTokenizer.from_pretrained(name)
text_encoder = T5EncoderModel.from_pretrained(name).eval()

tokenized = text_tokenizer.batch_encode_plus(
    [annotation_str],
    return_tensors="pt",
    padding="longest",
)
decoded_texts = [text_tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tokenized.input_ids]
