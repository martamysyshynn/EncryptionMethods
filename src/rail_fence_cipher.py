def fence_cipher_encrypt(text, height):
    if height <= 1 or height >= len(text):
        return text

    lines = []
    for i in range(height):
        lines.append('')

    row = 0
    step = 1

    for char in text:
        lines[row] += char
        if row == 0:
            step = 1
        elif row == height - 1:
            step = -1
        row += step

    return  ''.join(lines[::-1])

def fence_cipher_encrypt_by_blocks(text, height, block_length):
    if block_length == 0:
        return fence_cipher_encrypt(text, height)
    
    e_blocks = []
    for i in range(0, len(text), block_length):
        block = text[i:i+block_length]
        e_block = fence_cipher_encrypt(block, height)
        e_blocks.append(e_block)

    return ''.join(e_blocks)

def fence_cipher_decrypt(cipher_text, height):
    if height <= 1 or height >= len(cipher_text):
        return cipher_text
    
    lenght = [] 
    for x in range(height):
        lenght.append(0)

    row = 0
    step = 1

    for i in cipher_text:
        lenght[row] += 1
        if row == 0:
            step = 1
        elif row == height - 1:
            step = -1
        row += step

    lenght = lenght[::-1]

    lines = []
    index = 0
    for i in lenght:
        lines.append(list(cipher_text[index:index + i]))
        index += i

    lines = lines[::-1]

    source_text = []
    row = 0
    step = 1
    for j in cipher_text:
        source_text.append(lines[row].pop(0))
        if row == 0:
            step = 1
        elif row == height - 1:
            step = -1
        row += step

    return ''.join(source_text)

def fence_cipher_decrypt_by_blocks(chipher_text, height, block_length):
    if block_length == 0:
        return fence_cipher_decrypt(chipher_text, height)
    
    d_blocks = []
    for i in range(0, len(chipher_text), block_length):
        block = chipher_text[i:i+block_length]
        d_block = fence_cipher_decrypt(block, height)
        d_blocks.append(d_block)

    return ''.join(d_blocks)