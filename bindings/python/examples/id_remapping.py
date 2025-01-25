def test_id_remapping():
    """Example showing how to use the ID remapping processor"""
    from tokenizers import Tokenizer, processors
    from tokenizers.models import WordLevel


    # Initialize a simple tokenizer
    tokenizer = Tokenizer(WordLevel(vocab={"<unk>": 0, "a": 1, "b": 2, "c": 3},unk_token="<unk>"))

    # Create a mapping from some token IDs to new IDs
    id_map = {1: 100, 3: 300}

    # Create and set the ID remapping processor
    processor = processors.IdRemappingProcessor(id_map)
    tokenizer.post_processor = processor

    # The processor will remap IDs according to the mapping
    # IDs not in the mapping will remain unchanged
    print("ID remapping example:")
    output_tokens = tokenizer.encode(["a", "b", "c", "d" ], is_pretokenized=True)
    decoded_tokens = tokenizer.decode(output_tokens.ids)
    print(f"Original IDs: {output_tokens.ids}")  # Will show remapped IDs where applicable
    print(f"Decoded tokens: {decoded_tokens}")


if __name__ == "__main__":
    test_id_remapping()
