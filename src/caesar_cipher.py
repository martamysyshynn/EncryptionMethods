from collections import Counter

def caesars_cipher_encrypt(plain_text, alphabet, key):
    cipher_text = ''
    for i in plain_text:
        if i in alphabet:
            index = alphabet.index(i)
            new_index = (index + key) % len(alphabet)
            cipher_text += alphabet[new_index]
        elif i.upper() in alphabet:
            index = alphabet.index(i.upper())
            new_index = (index + key) % len(alphabet)
            cipher_text += alphabet[new_index].upper() if i.isupper() else alphabet[new_index]
        else:
            cipher_text += i
    return cipher_text

def caesars_cipher_encrypt_by_blocks(text, block_length, alphabet, key):
    if block_length == 0:
        return caesars_cipher_encrypt(text, alphabet, key)
    
    encrypted_text = ''
    for i in range(0, len(text), block_length):
        block = text[i:i + block_length]
        encrypted_block = caesars_cipher_encrypt(block, alphabet, key)
        encrypted_text += encrypted_block

    return encrypted_text 

def caesars_cipher_decrypt(text, alphabet, key):
    return caesars_cipher_encrypt(text, alphabet, -key)

def caesars_cipher_decrypt_by_blocks(text, block_length, alphabet, key):
    if block_length == 0:
        return caesars_cipher_decrypt(text, alphabet, key)
    
    decrypted_text = ''
    for i in range(0, len(text), block_length):
        block = text[i:i + block_length]
        decrypted_block = caesars_cipher_decrypt(block, alphabet, key)
        decrypted_text += decrypted_block

    return decrypted_text

def frequency_analysis(text, alphabet, file_name):
    filtered_text = [char for char in text if char in alphabet]

    text_counter = Counter(filtered_text)
    if not text_counter:
        return None
    
    total_chars = len(filtered_text)
    sorted_letters = text_counter.most_common()

    with open(file_name, 'w', encoding='utf-8') as file:
        for letter, freq in sorted_letters:
            percentage = (freq / total_chars) * 100
            file.write(f"{letter}: {percentage:.2f}%\n")

    most_common_letter, _ = sorted_letters[0]

    reference_letter = 'E' if 'E' in alphabet else 'О'

    if most_common_letter in alphabet:
        key = (alphabet.index(most_common_letter) - alphabet.index(reference_letter)) % len(alphabet)
        return key
    return None

def caesars_cipher_decrypt_by_frequency_analysis(text, alphabet, file_name):
    key = frequency_analysis(text, alphabet, file_name)

    if key is None:
        return None
    
    return caesars_cipher_decrypt(text, alphabet, key)

