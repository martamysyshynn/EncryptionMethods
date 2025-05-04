import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import tracemalloc
from collections import Counter
import random
from abc import ABC, abstractmethod
import math
from math import gcd
from functools import reduce
from string import ascii_uppercase
import numpy as np

class Cipher(ABC):

    @abstractmethod
    def encrypt(self, text):
        pass

    @abstractmethod
    def decrypt(self, text):
        pass
    
    @staticmethod
    @abstractmethod
    def get_widgets(frame):
        pass

    @staticmethod
    @abstractmethod
    def process_block(text, block_length, encrypt=True):
        pass
    

class RailFenceCipher(Cipher):

    def __init__(self, height):
        self._height = height
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        if not isinstance(value, int):
            raise ValueError("Height must be an integer")
        self._height = value

    def encrypt(self, plain_text):
        if self._height <= 1 or self._height >= len(plain_text):
            return plain_text

        lines = []
        for i in range(self._height):
            lines.append('')

        row = 0
        step = 1

        for char in plain_text:
            lines[row] += char
            if row == 0:
                step = 1
            elif row == self._height - 1:
                step = -1
            row += step

        return  ''.join(lines[::-1])
    
    def decrypt(self, cipher_text):
        if self._height <= 1 or self._height >= len(cipher_text):
            return cipher_text
        
        lenght = [] 
        for x in range(self._height):
            lenght.append(0)

        row = 0
        step = 1

        for i in cipher_text:
            lenght[row] += 1
            if row == 0:
                step = 1
            elif row == self._height - 1:
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
            elif row == self._height - 1:
                step = -1
            row += step

        return ''.join(source_text)
    
    @staticmethod
    def get_widgets(frame):
        widgets = {}
        
        widgets['height_label'] = tk.Label(frame, text='Height: ')
        widgets['height_entry'] = tk.Entry(frame, width=10)
        
        widgets['block_length_label'] = tk.Label(frame, text='Block length: ')
        widgets['block_length_entry'] = tk.Entry(frame, width=10)
        
        return widgets
    
    @staticmethod
    def process_block(text, block_length, encrypt=True, **kwargs):
        if block_length == 0:
            if encrypt:
                return kwargs['cipher'].encrypt(text)
            return kwargs['cipher'].decrypt(text)
    
        processed_text = []
        for i in range(0, len(text), block_length):
            block = text[i:i+block_length]
            if encrypt:
                processed_block = kwargs['cipher'].encrypt(block)
            else:
                processed_block = kwargs['cipher'].decrypt(block)
            processed_text.append(processed_block)
        return ''.join(processed_text)
    

class CaesarCipher(Cipher):

    def __init__(self, key, alphabet):
        self._key = key
        self._alphabet = alphabet.upper() if alphabet else None

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        if not isinstance(value, int):
            raise ValueError("Key must be an integer")
        self._key = value

    @property
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, value):
        if not isinstance(value, str):
            raise ValueError("Alphabet must be a string")
        self._alphabet = value.upper() 

    def encrypt(self, plain_text):
        return self._caesar_shift(plain_text, self.key)
        
    def decrypt(self, cipher_text):
        return self._caesar_shift(cipher_text, -self.key)
    
    def _caesar_shift(self, text, shift):
        result = []
        for i in text: 
            if i in self.alphabet:
                index = self.alphabet.index(i)
                new_index = (index + shift) % len(self.alphabet)
                result.append(self.alphabet[new_index])
            elif i.upper() in self.alphabet:
                index = self.alphabet.index(i.upper())
                new_index = (index + shift) % len(self.alphabet)
                result.append(self.alphabet[new_index].upper() if i.isupper() else self.alphabet[new_index])
            else:
                result.append(i)
        return ''.join(result)
    
    @staticmethod
    def get_widgets(frame):
        widgets = {}
        
        widgets['shear_label'] = tk.Label(frame, text='Shear value: ')
        widgets['shear_value'] = tk.Entry(frame, width=10)
        
        widgets['alphabet_label'] = tk.Label(frame, text='Alphabet file: ')
        widgets['alphabet_entry'] = tk.Entry(frame, width=50)
        widgets['alphabet_button'] = tk.Button(frame, text='Browse')
        
        widgets['block_length_label'] = tk.Label(frame, text='Block length: ')
        widgets['block_length_entry'] = tk.Entry(frame, width=10)
        
        widgets['frequency_button'] = tk.Button(frame, text='Frequency Analysis')

        return widgets
    
    @staticmethod
    def process_block(text, block_length, encrypt=True, **kwargs):
        if block_length == 0:
            if encrypt:
                return kwargs['cipher'].encrypt(text)
            return kwargs['cipher'].decrypt(text)
    
        processed_text = []
        for i in range(0, len(text), block_length):
            block = text[i:i + block_length]
            if encrypt:
                processed_block = kwargs['cipher'].encrypt(block)
            else:
                processed_block = kwargs['cipher'].decrypt(block)
            processed_text.append(processed_block)
        return ''.join(processed_text)
    
    @staticmethod
    def frequency_analysis(text, alphabet, output_file):
        
        filtered_text = [char for char in text if char in alphabet]
        if not filtered_text:
            return None
            
        text_counter = Counter(filtered_text)
        total_chars = len(filtered_text)
        sorted_letters = text_counter.most_common()
        
        with open(output_file, 'w', encoding='utf-8') as file:
            for letter, freq in sorted_letters:
                percentage = (freq / total_chars) * 100
                file.write(f"{letter}: {percentage:.2f}%\n")
                
        most_common_letter, _ = sorted_letters[0]
        reference_letter = 'E' if 'E' in alphabet else 'О'
        
        if most_common_letter in alphabet:
            key = (alphabet.index(most_common_letter) - alphabet.index(reference_letter)) % len(alphabet)
            return key
        return None
    
class CardanoCipher(Cipher):

    def __init__(self, key=None, size=None):
        self.key = key or []
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if not isinstance(value, int) or value % 2 != 0:
            raise ValueError("Size must be an even integer")
        self._size = value

    def encrypt(self, plain_text):
        block_size = self.size * self.size
        if len(plain_text) > block_size:
            return self._encrypt_by_blocks(plain_text)
            
        plain_text = plain_text.replace(' ', '').ljust(block_size, '*')
        grid = [[' ' for i in range(self.size)] for j in range(self.size)]
        
        key_grid = self._create_key_grid()
        
        index = 0
        for x in range(4):
            for row in range(self.size):
                for col in range(self.size):
                    if key_grid[row][col] == 'X' and index < len(plain_text):
                        grid[row][col] = plain_text[index]
                        index += 1
            key_grid = self._rotate_matrix(key_grid)
            
        return ''.join(''.join(row) for row in grid)
    
    def decrypt(self, encrypted_text):
        block_size = self.size * self.size
        if len(encrypted_text) > block_size:
            return self._decrypt_by_blocks(encrypted_text)
            
        grid = self._text_to_grid(encrypted_text)
        key_grid = self._create_key_grid()
        
        decrypted_text = []
        for x in range(4):
            for row in range(self.size):
                for col in range(self.size):
                    if key_grid[row][col] == 'X':
                        decrypted_text.append(grid[row][col])
            key_grid = self._rotate_matrix(key_grid)
            
        return ''.join(decrypted_text).rstrip('*')
    
    def _encrypt_by_blocks(self, text):
        block_size = self.size * self.size
        encrypted_blocks = []
        i = 0
        while i < len(text):
            block = text[i:i + block_size]
            encrypted_blocks.append(self.encrypt(block))
            i += block_size
        return ''.join(encrypted_blocks)
        
    def _decrypt_by_blocks(self, encrypted_text):
        block_size = self.size * self.size
        decrypted_blocks = []
        i = 0
        while i < len(encrypted_text):
            block = encrypted_text[i:i + block_size]
            decrypted_blocks.append(self.decrypt(block))
            i += block_size
        return ''.join(decrypted_blocks)
        
    def _create_key_grid(self):
        key_grid = [[' ' for i in range(self.size)] for j in range(self.size)]
        for row, col in self.key:
            key_grid[row][col] = 'X'
        return key_grid
        
    def _text_to_grid(self, text):
        grid = [[' ' for i in range(self.size)] for j in range(self.size)]
        index = 0
        for row in range(self.size):
            for col in range(self.size):
                if index < len(text):
                    grid[row][col] = text[index]
                    index += 1
        return grid
        
    @staticmethod
    def _rotate_matrix(matrix):
        return list(zip(*matrix[::-1]))
    
    @staticmethod
    def get_widgets(frame):
        widgets = {}
        
        widgets['key_size_label'] = tk.Label(frame, text='Key size: ')
        widgets['key_size_entry'] = tk.Entry(frame)
        widgets['key_size_button'] = tk.Button(frame, text='Set key size')
        
        widgets['generate_key'] = tk.Button(frame, text='Generate Random Key')
        widgets['key_import_from_file'] = tk.Button(frame, text='Import key')
        widgets['key_export_to_file'] = tk.Button(frame, text='Export key')
        
        widgets['cells_canvas'] = tk.Canvas(frame)
        
        return widgets
    
    @staticmethod
    def process_block(text, block_length, encrypt=True, **kwargs):
        if encrypt:
            return kwargs['cipher'].encrypt(text)
        return kwargs['cipher'].decrypt(text)
    
class VigenereCipher(Cipher):
    def __init__(self, key=None, alphabet=None):
        self._key = key
        self._alphabet = alphabet.upper() if alphabet else None

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, value):
        if not isinstance(value, str):
            raise ValueError("Key must be an string")
        self._key = value

    @property
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, value):
        if not isinstance(value, str):
            raise ValueError("Alphabet must be a string")
        self._alphabet = value.upper() 

    def encrypt(self, plain_text):
        return self.vigenere_shift(plain_text, self.key, decrypt=False)
        
    def decrypt(self, cipher_text):
        return self.vigenere_shift(cipher_text, self.key, decrypt=True)

    def vigenere_shift(self, text, keyword, decrypt=False):
        result = []
        key = keyword.upper()
        key_lenght = len(key)
        key_index = 0

        for i in text: 
            if i.upper() in self.alphabet:
                shift_letter = key[key_index % key_lenght]
                shift = self.alphabet.index(shift_letter)

                if decrypt:
                    shift = -shift
                if i in self.alphabet:
                    index = self.alphabet.index(i)
                    new_index = (index + shift) % len(self.alphabet)
                    result.append(self.alphabet[new_index])
                else:
                    index = self.alphabet.index(i.upper())
                    new_index = (index + shift) % len(self.alphabet)
                    result.append(self.alphabet[new_index].upper() if i.isupper() else self.alphabet[new_index])
                key_index += 1
            else:
                result.append(i)
        return ''.join(result)
    
    
    @staticmethod
    def get_widgets(frame):
        widgets = {}
        
        widgets['keyword_label'] = tk.Label(frame, text='Keyword: ')
        widgets['keyword_value'] = tk.Entry(frame, width=10)
        
        widgets['alphabet_label'] = tk.Label(frame, text='Alphabet file: ')
        widgets['alphabet_entry'] = tk.Entry(frame, width=50)
        widgets['alphabet_button'] = tk.Button(frame, text='Browse')
        widgets['block_length_label'] = tk.Label(frame, text='Block length: ')
        widgets['block_length_entry'] = tk.Entry(frame, width=10)
        widgets['kasiski_button'] = tk.Button(frame, text='Kasiski Algorithm')

        widgets['brute_force_button'] = tk.Button(frame, text='Brute Force (Dictionary)')
        return widgets
    
    @staticmethod
    def process_block(text, block_length, encrypt=True, **kwargs):
        cipher = kwargs['cipher']
        if block_length == 0:
            if encrypt:
                return cipher.encrypt(text)
            return cipher.decrypt(text)
    
        processed_text = []
        for i in range(0, len(text), block_length):
            block = text[i:i + block_length]
            if encrypt:
                processed_block = cipher.encrypt(block)
            else:
                processed_block = cipher.decrypt(block)
            processed_text.append(processed_block)
        return ''.join(processed_text)
    
    @staticmethod
    def kasiski_algorithm(cipher_text, output_file=None):
        cipher_text = ''.join(char for char in cipher_text if char.isalpha())

        sequences = {}
        min_seq_length = 3
        n = len(cipher_text)
        for i in range(n - min_seq_length + 1):
            seq = cipher_text[i:i + min_seq_length]
            if seq not in sequences:
                sequences[seq] = []
            sequences[seq].append(i)

        repeated_sequences = {seq: pos for seq, pos in sequences.items() if len(pos) >= 2}
        if not repeated_sequences:
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("No repeated sequences of length 3 found.\n")
            return None

        distances = []
        for seq, positions in repeated_sequences.items():
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    distances.append(positions[j] - positions[i])

        if not distances:
            if output_file:
                with open(output_file, 'w') as f:
                    f.write("No distances calculated.\n")
            return None
        
        def gcd_all(lst):
            return reduce(gcd, lst)

        overall_gcd = gcd_all(distances)
        likely_key_length = overall_gcd

        if output_file:
            with open(output_file, 'w') as f:
                f.write("Kasiski Analysis Report\n")
                f.write("=======================\n\n")
                f.write(f"Most Likely Keyword Length (GCD of distances): {likely_key_length}\n")
                f.write("\nPossible Divisors (Actual Key Length Candidates):\n")
                if likely_key_length > 1:
                    divisors = [d for d in range(2, likely_key_length + 1) if likely_key_length % d == 0]
                    for d in divisors:
                        f.write(f"- {d}\n")
                else:
                    f.write("No valid divisors found.\n")

        return likely_key_length
        
    @staticmethod
    def brute_force(cipher_text, dictionary_path, output_file=None, alphabet=ascii_uppercase):
        with open(dictionary_path, 'r') as f:
            english_words = [word.strip().upper() for word in f.readlines()]
        english_set = set(english_words)  

        new_score = -1
        new_key = None
        new_decrypted = ""
        
        for swelling_key in english_words:
            if not swelling_key:
                continue

            cipher = VigenereCipher(key=swelling_key, alphabet=alphabet)
            decrypted = cipher.decrypt(cipher_text)
         
            words = decrypted.split()
            score = 0
            for word in words:
                clean_word = ''.join([c for c in word if c.isalpha()]).upper()
                if clean_word in english_set:
                    score += 1
            
            if (score > new_score) or (score == new_score and len(swelling_key) < len(new_key)):
                new_score = score
                new_key = swelling_key
                new_decrypted = decrypted
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Brute Force Attack Results\n")
                f.write("==========================\n\n")
                f.write(f"Best Candidate Key: {new_key}\n")
                f.write(f"Matches Found: {new_score}\n")
        
        return (new_key, new_decrypted)

class AffineCipher(Cipher):
    def __init__(self, key_matrix, shift_vector, alphabet):
        self.key_matrix = np.array(key_matrix, dtype=int) 
        self.shift_vector = np.array(shift_vector, dtype=int)
        self.matrix_size = self.key_matrix.shape[0] 
        self.alphabet = alphabet.upper()  
        self.modulus = len(self.alphabet)
         
        
        if self.key_matrix.shape[0] != self.key_matrix.shape[1]:
            raise ValueError("The matrix must be square")

        det = int(round(np.linalg.det(self.key_matrix))) % self.modulus
        if math.gcd(det, self.modulus) != 1:
            raise ValueError("Matrix determinant must be coprime with modulus")
        
    def encrypt(self, text):
        vectors = self.text_to_vectors(text)
        encrypted_vectors = []
        for vec in vectors:
            encrypted = (np.dot(self.key_matrix, vec) + self.shift_vector) % self.modulus
            encrypted_vectors.append(encrypted)
        return self.vectors_to_text(encrypted_vectors)
    
    def decrypt(self, text):
        inverse_matrix = self.matrix_inversion(self.key_matrix, self.modulus)
        vectors = self.text_to_vectors(text)
        decrypted_vectors = []
        for vec in vectors:
            shifted = (vec - self.shift_vector) % self.modulus
            decrypted = np.dot(inverse_matrix, shifted) % self.modulus
            decrypted_vectors.append(decrypted)
        return self.vectors_to_text(decrypted_vectors)
    
    @staticmethod
    def get_widgets(frame):
        widgets = {}
        
        widgets['matrix_size_label'] = tk.Label(frame, text='Matrix size: ')
        widgets['matrix_size_entry'] = tk.Entry(frame)

        widgets['matrix_label'] = tk.Label(frame, text='Matrix value: ')
        widgets['matrix_entry'] = tk.Entry(frame)

        widgets['vector_label'] = tk.Label(frame, text='Shift vector:')
        widgets['vector_entry'] = tk.Entry(frame)

        widgets['alphabet_label'] = tk.Label(frame, text='Alphabet file: ')
        widgets['alphabet_entry'] = tk.Entry(frame, width=50)
        widgets['alphabet_button'] = tk.Button(frame, text='Browse')

        widgets['block_length_label'] = tk.Label(frame, text='Block length: ')
        widgets['block_length_entry'] = tk.Entry(frame, width=10)

        widgets['generate_key_button'] = tk.Button(frame, text='Generate Random Key')

        widgets['import_key'] = tk.Button(frame, text='Import key')
        widgets['export_key'] = tk.Button(frame, text='Export key')

        return widgets

    @staticmethod
    def matrix_inversion(matrix, modulus):
        """Return the modular inverse of a matrix under a given modulus."""
        det = int(round(np.linalg.det(matrix))) % modulus
        if det == 0 or math.gcd(det, modulus) != 1:
            raise ValueError("Matrix is not invertible under this modulus.")
        
        n = matrix.shape[0]
        cofactors = np.zeros((n, n), dtype=int)

        for row in range(n):
            for col in range(n):
                minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
                cofactor = int(round(np.linalg.det(minor)))
                sign = (-1) ** (row + col)
                cofactors[row, col] = sign * cofactor

        transpose  = cofactors.T % modulus

        det_inv = pow(det, -1, modulus)
        inverse_matrix = (det_inv * transpose) % modulus

        return inverse_matrix

    def text_to_vectors(self, text):
        filtered_text = [c.upper() for c in text if c.upper() in self.alphabet]
        text = ''.join(filtered_text)

        padding = (-len(text)) % self.matrix_size
        text += self.alphabet[0] * padding
        
        vectors = []
        for i in range(0, len(text), self.matrix_size):
            block = text[i:i+self.matrix_size]
            vec = [self.alphabet.index(c) for c in block]
            vectors.append(np.array(vec))
        return vectors

    def vectors_to_text(self, vectors):
        text = []
        for vec in vectors:
            for num in vec:
                text.append(self.alphabet[int(num) % self.modulus])
        return "".join(text)
    
    @staticmethod
    def process_block(text, block_length, encrypt=True, **kwargs):
        cipher = kwargs['cipher']
        matrix_size = cipher.matrix_size
        modulus = cipher.modulus
        alphabet = cipher.alphabet

        if block_length == 0:
            return cipher.encrypt(text) if encrypt else cipher.decrypt(text)
        
        if block_length % matrix_size != 0:
            raise ValueError("Block length must be a multiple of matrix size")

        processed_blocks = []
        for i in range(0, len(text), block_length):
            block = text[i:i + block_length]
            if encrypt and len(block) < block_length:
                padding = alphabet[0] * (block_length - len(block))
                block += padding
            if encrypt:
                processed_block = cipher.encrypt(block)
            else:
                processed_block = cipher.decrypt(block)
            processed_blocks.append(processed_block)

        result = ''.join(processed_blocks)
        if not encrypt:
            result = result[:len(text)]
        return result
            
class CipherApp:

    def __init__(self, root):
        self.root = root
        self.root.title('Encryption & Decryption')
        self.root.geometry('1200x800')
        
        self.selected_cipher = tk.StringVar(value="cardano")
        self.encrypt_var = tk.BooleanVar(value=True)
        self.selected_cells = []
        
        self.ciphers = {
            "cardano": CardanoCipher,
            "rail_fence": RailFenceCipher,
            "caesar": CaesarCipher,
            "vigenere": VigenereCipher,
            "affine": AffineCipher
        }
        
        self.current_cipher = None
        self.cipher_widgets = {}
        
        self.create_widgets()
        self.toggle_cipher_options()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.frame = ttk.Frame(self.canvas)
        
        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
  
        cipher_frame = ttk.LabelFrame(self.frame, text="Select Cipher")
        cipher_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        
        for cipher_name in self.ciphers:
            tk.Radiobutton(
                cipher_frame, 
                text=f"{cipher_name.replace('_', ' ').title() + ' Cipher'}",
                variable=self.selected_cipher,
                value=cipher_name,
                command=self.toggle_cipher_options
            ).pack(side=tk.LEFT)
        
        mode_frame = ttk.LabelFrame(self.frame, text="Operation Mode")
        mode_frame.grid(row=0, column=3, columnspan=3, padx=5, pady=5, sticky="w")
        
        tk.Radiobutton(
            mode_frame, 
            text="Encrypt", 
            variable=self.encrypt_var, 
            value=True,
            command=lambda: [self.toggle_cipher_options(), self.frequency_button.grid_remove()]
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            mode_frame, 
            text="Decrypt", 
            variable=self.encrypt_var, 
            value=False,
            command=self.toggle_cipher_options
        ).pack(side=tk.LEFT)
        
        self.input_label = tk.Label(self.frame, text='Input file: ')
        self.input_entry = tk.Entry(self.frame, width=50)
        self.input_button = tk.Button(self.frame, text='Browse', command=lambda: self.search_file(self.input_entry))
        
        self.output_label = tk.Label(self.frame, text='Output file: ')
        self.output_entry = tk.Entry(self.frame, width=50)
        self.output_button = tk.Button(self.frame, text='Browse', command=lambda: self.search_file(self.output_entry))
        
        self.res_text = tk.Text(self.frame, height=10, width=70)
        self.res_text.grid(row=14, column=0, columnspan=3)
      
        self.time_label = tk.Label(self.frame, text="Time taken: ")
        self.time_label.grid(row=15, column=0, columnspan=3)
        
        self.memory_label = tk.Label(self.frame, text="Memory usage: ")
        self.memory_label.grid(row=16, column=0, columnspan=3)
  
        self.process_button = tk.Button(self.frame, text='Process', command=self.process)

        self.input_label.grid(row=1, column=0, padx=5, pady=5)
        self.input_entry.grid(row=1, column=1, padx=5, pady=5)
        self.input_button.grid(row=1, column=2, padx=5, pady=5)
        
        self.output_label.grid(row=2, column=0, padx=5, pady=5)
        self.output_entry.grid(row=2, column=1, padx=5, pady=5)
        self.output_button.grid(row=2, column=2, padx=5, pady=5)
        
        self.process_button.grid(row=13, column=0, columnspan=3, pady=10)
        
    def toggle_cipher_options(self):
        cipher_name = self.selected_cipher.get()
    
        for widget in self.cipher_widgets.values():
            widget.grid_remove()
        
        cipher_class = self.ciphers[cipher_name]
        self.cipher_widgets = cipher_class.get_widgets(self.frame)

        if cipher_name == "rail_fence":
            self.cipher_widgets['height_label'].grid(row=3, column=0, padx=5, pady=5)
            self.cipher_widgets['height_entry'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['block_length_label'].grid(row=4, column=0, padx=5, pady=5)
            self.cipher_widgets['block_length_entry'].grid(row=4, column=1, padx=5, pady=5, sticky="w")
            
        elif cipher_name == "caesar":
            self.cipher_widgets['shear_label'].grid(row=3, column=0, padx=5, pady=5)
            self.cipher_widgets['shear_value'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['alphabet_label'].grid(row=4, column=0, padx=5, pady=5)
            self.cipher_widgets['alphabet_entry'].grid(row=4, column=1, padx=5, pady=5)
            self.cipher_widgets['alphabet_button'].grid(row=4, column=2, padx=5, pady=5)
            self.cipher_widgets['block_length_label'].grid(row=5, column=0, padx=5, pady=5)
            self.cipher_widgets['block_length_entry'].grid(row=5, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['frequency_button'].config(command=self.perform_frequency_analysis)
            if not self.encrypt_var.get():
                self.cipher_widgets['frequency_button'].grid(row=6, column=0, columnspan=3, pady=5)
            else:
                self.cipher_widgets['frequency_button'].grid_remove()
                
        elif cipher_name == "cardano":
            self.cipher_widgets['key_size_label'].grid(row=3, column=0, padx=5, pady=5)
            self.cipher_widgets['key_size_entry'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['key_size_button'].grid(row=3, column=2, padx=5, pady=5)
            self.cipher_widgets['generate_key'].grid(row=4, column=0, columnspan=3, pady=5)
            self.cipher_widgets['key_import_from_file'].grid(row=5, column=0, columnspan=3, pady=5)
            self.cipher_widgets['key_export_to_file'].grid(row=6, column=0, columnspan=3, pady=5)
            self.cipher_widgets['cells_canvas'].grid(row=7, column=0, columnspan=3, pady=10)
            
            self.cipher_widgets['key_size_button'].config(command=self.get_size_key)
            self.cipher_widgets['generate_key'].config(command=self.generate_random_key)
            self.cipher_widgets['key_import_from_file'].config(command=self.import_key_from_file)
            self.cipher_widgets['key_export_to_file'].config(command=self.export_key_to_file)

            if 'alphabet_button' in self.cipher_widgets:
                self.cipher_widgets['alphabet_button'].config(command=lambda: self.search_file(self.cipher_widgets['alphabet_entry']))
        
        elif cipher_name == 'vigenere':
            self.cipher_widgets['keyword_label'].grid(row=3, column=0, padx=5, pady=5)
            self.cipher_widgets['keyword_value'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['alphabet_label'].grid(row=4, column=0, padx=5, pady=5)
            self.cipher_widgets['alphabet_entry'].grid(row=4, column=1, padx=5, pady=5)
            self.cipher_widgets['alphabet_button'].grid(row=4, column=2, padx=5, pady=5)
            self.cipher_widgets['block_length_label'].grid(row=5, column=0, padx=5, pady=5)
            self.cipher_widgets['block_length_entry'].grid(row=5, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['kasiski_button'].config(command=self.perform_kasiski_analysis)
            if not self.encrypt_var.get():
                self.cipher_widgets['kasiski_button'].grid(row=6, column=0, columnspan=3, pady=5)
            else:
                self.cipher_widgets['kasiski_button'].grid_remove()

            self.cipher_widgets['brute_force_button'].config(command=self.perform_brute_force)
            if not self.encrypt_var.get():
                self.cipher_widgets['brute_force_button'].grid(row=7, column=0, columnspan=3, pady=5)
            else:
                self.cipher_widgets['brute_force_button'].grid_remove()

        elif cipher_name == 'affine':
            self.cipher_widgets['matrix_size_label'].grid(row=3, column=0, padx=5, pady=5)
            self.cipher_widgets['matrix_size_entry'].grid(row=3, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['matrix_label'].grid(row=4, column=0, padx=5, pady=5)
            self.cipher_widgets['matrix_entry'].grid(row=4, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['vector_label'].grid(row=5, column=0, padx=5, pady=5)
            self.cipher_widgets['vector_entry'].grid(row=5, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['alphabet_label'].grid(row=6, column=0, padx=5, pady=5)
            self.cipher_widgets['alphabet_entry'].grid(row=6, column=1, padx=5, pady=5)
            self.cipher_widgets['alphabet_button'].grid(row=6, column=2, padx=5, pady=5)
            self.cipher_widgets['block_length_label'].grid(row=7, column=0, padx=5, pady=5)
            self.cipher_widgets['block_length_entry'].grid(row=7, column=1, padx=5, pady=5, sticky="w")
            self.cipher_widgets['generate_key_button'].grid(row=8, column=0, columnspan=3, pady=5)
            self.cipher_widgets['generate_key_button'].config(command=self.generate_key_for_affine_cipher)
            self.cipher_widgets['import_key'].grid(row=9, column=0, columnspan=2, pady=5)
            self.cipher_widgets['import_key'].config(command=self.import_key_for_affine)
            self.cipher_widgets['export_key'].grid(row=9, column=1, columnspan=3, pady=5)
            self.cipher_widgets['export_key'].config(command=self.export_key_for_affine)

    def _get_block_length(self):
        cipher_name = self.selected_cipher.get()
        if cipher_name == "caesar":
            return int(self.cipher_widgets['block_length_entry'].get())
        elif cipher_name == "rail_fence":
            return int(self.cipher_widgets['block_length_entry'].get())
        elif cipher_name == "vigenere":
            entry_value = self.cipher_widgets['block_length_entry'].get()
            return int(entry_value) if entry_value.isdigit() else 0
        elif cipher_name == "affine":
            entry_value = self.cipher_widgets['block_length_entry'].get()
            return int(entry_value) if entry_value.isdigit() else 0 
        return 0
    
    def perform_frequency_analysis(self):
        if self.selected_cipher.get() != "caesar" or self.encrypt_var.get():
            return
        
        input_file = self.input_entry.get()
        if not input_file:
            self.analysis_label.config(text="Error: No input file specified")
            return

        alphabet_file = self.cipher_widgets['alphabet_entry'].get()
        if not alphabet_file:
            self.analysis_label.config(text="Error: No alphabet file specified")
            return
        
        try:
            ciphertext = self.read_file(input_file)
            alphabet = self.read_file(alphabet_file)
            
            if not ciphertext:
                self.analysis_label.config(text="Error: Input file is empty")
                return
            if not alphabet:
                self.analysis_label.config(text="Error: Alphabet file is empty")
                return
            
            output_file = "frequency_analysis.txt"
            key = CaesarCipher.frequency_analysis(ciphertext, alphabet, output_file)
            
            if key is not None:
                try:
                    self.cipher_widgets['shear_value'].delete(0, tk.END)
                    self.cipher_widgets['shear_value'].insert(0, str(key))
                    
                    block_length = int(self.cipher_widgets['block_length_entry'].get())
                    cipher = CaesarCipher(key, alphabet)
                    decrypted_text = cipher.decrypt(ciphertext)
                    
                    output_path = self.output_entry.get()
                    if output_path:
                        self.write_to_file(output_path, decrypted_text)
                    
                    self.res_text.delete("1.0", tk.END)
                    self.res_text.insert(tk.END, decrypted_text)
                    
                    self.analysis_label.config(text=
                        f"Frequency analysis complete!\n"
                        f"Most likely key: {key}\n"
                        f"Results saved to: {output_file}"
                    )
                except ValueError as e:
                    self.analysis_label.config(text=f"Error: Invalid block length - {str(e)}")
                except Exception as e:
                    self.analysis_label.config(text=f"Error during decryption: {str(e)}")
            else:
                self.analysis_label.config(text=
                    "Frequency analysis couldn't determine the key.\n"
                    "Check the file frequency_analysis.txt for character frequencies."
                )
        except Exception as e:
            self.analysis_label.config(text=f"Error during frequency analysis: {str(e)}")
    
    def get_size_key(self):
        key_size = self.cipher_widgets['key_size_entry'].get()
        if key_size.isdigit() and int(key_size) >= 2:
            key_size = int(key_size)
            if key_size % 2 == 0:
                self.create_matrix(key_size)
            else:
                messagebox.showerror("Error", "Invalid size. Enter an even number.")
        else:
            messagebox.showerror("Error", "Invalid input. Enter an even number")
    
    def create_matrix(self, size):
        canvas = self.cipher_widgets['cells_canvas']
        canvas.delete('all')
        cell_size = max(5, min(40, 800 // size))
        canvas.config(width=size*cell_size, height=size*cell_size)

        for row in range(size):
            for col in range(size):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                canvas.create_rectangle(
                    x1, y1, x2, y2, 
                    fill='white', 
                    outline='pink', 
                    tags=f'cell_{row}_{col}'
                )
                canvas.tag_bind(
                    f"cell_{row}_{col}", 
                    "<Button-1>", 
                    lambda e, row=row, col=col: self.select_cell(row, col)
                )
    
    def rotate_position(self, positions, size):
        return [(col, size - 1 - row) for row, col in positions]
    
    def forbidden_cells(self, selected_cells, size):
        forbidden = set()
        for cell in selected_cells:
            rotated = [cell]
            for x in range(3):
                rotated.append(self.rotate_position([rotated[-1]], size)[0])
            forbidden.update(rotated)
        return forbidden
    
    def update_colors(self):
        size = int(self.cipher_widgets['key_size_entry'].get())
        forbidden_cells = self.forbidden_cells(self.selected_cells, size)
        canvas = self.cipher_widgets['cells_canvas']

        for row in range(size):
            for col in range(size):
                cell_id = f'cell_{row}_{col}'
                if (row, col) in self.selected_cells:
                    canvas.itemconfig(cell_id, fill='pink')
                elif (row, col) in forbidden_cells:
                    canvas.itemconfig(cell_id, fill='darkgrey') 
                else:
                    canvas.itemconfig(cell_id, fill='white')    
    
    def select_cell(self, row, col):
        size = int(self.cipher_widgets['key_size_entry'].get())
        if (row, col) in self.selected_cells:
            self.selected_cells.remove((row, col))
        else:
            if (row, col) not in self.forbidden_cells(self.selected_cells, size):
                self.selected_cells.append((row, col))
        self.update_colors()
        
    def generate_random_key(self):
        key_size = self.cipher_widgets['key_size_entry'].get()
        if not key_size.isdigit() or int(key_size) % 2 != 0:
            messagebox.showerror("Error", "Invalid key size. Please enter an even number.")
            return
        size = int(key_size)

        if not self.cipher_widgets['cells_canvas'].find_all(): 
            self.create_matrix(size)

        self.selected_cells.clear()
        forbidden = set()
        
        positions = [(row, col) for row in range(size) for col in range(size)]
        random.shuffle(positions) 
        
        for row, col in positions:
            if (row, col) not in forbidden:
                rotations = []
                current_row, current_col = row, col
                for _ in range(4):
                    rotations.append((current_row, current_col))
                    current_row, current_col = current_col, size - 1 - current_row

                if any(pos in forbidden for pos in rotations):
                    continue

                selected = random.choice(rotations)
                self.selected_cells.append(selected)
                forbidden.update(rotations)
        
        self.update_colors()    

    def import_key_from_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Key File", 
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    key_data = file.read().strip().splitlines()
                key_size = int(key_data[0])

                self.cipher_widgets['key_size_entry'].delete(0, tk.END)
                self.cipher_widgets['key_size_entry'].insert(0, str(key_size))
                
                self.create_matrix(key_size) 
                self.selected_cells.clear()
                for i in key_data[1:]:
                    row, col = map(int, i.split(','))
                    self.selected_cells.append((row, col))
                    cell_id = f"cell_{row}_{col}"
                    self.cipher_widgets['cells_canvas'].itemconfig(cell_id, fill='pink')
            except Exception as e:
                messagebox.showerror("Error", f"Error opening file with key: {e}")
    
    def export_key_to_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    key_size = int(self.cipher_widgets['key_size_entry'].get())
                    file.write(f'{key_size}\n')
                    for row, col in self.selected_cells:
                        file.write(f'{row},{col}\n')
                messagebox.showinfo("Success", "The key has been exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting key: {e}")
    
    def perform_kasiski_analysis(self):
        if self.selected_cipher.get() != "vigenere" or self.encrypt_var.get():
            return
        
        input_file = self.input_entry.get()
        if not input_file:
            messagebox.showerror("Error", "Please select an input file.")
            return
        
        ciphertext = self.read_file(input_file)
        if not ciphertext:
            messagebox.showerror("Error", "The input file is empty.")
            return
        
        output_file = "kasiski_analysis.txt"
        
        try:
            key_length = VigenereCipher.kasiski_algorithm(ciphertext, output_file)
            if key_length is not None:
                messagebox.showinfo(
                    "Kasiski Analysis",
                    f"Most likely keyword length: {key_length}\nReport saved to {output_file}."
                )
            else:
                messagebox.showinfo(
                    "Kasiski Analysis",
                    f"Unable to determine keyword length.\nReport saved to {output_file}."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Error during Kasiski analysis: {str(e)}")

    def perform_brute_force(self):
        input_file = self.input_entry.get()
        alphabet_file = self.cipher_widgets['alphabet_entry'].get()
        
        if not input_file or not alphabet_file:
            messagebox.showerror("Error", "Missing input file or alphabet file!")
            return
        
        ciphertext = self.read_file(input_file)
        alphabet = self.read_file(alphabet_file)
        
        dict_file = filedialog.askopenfilename(
            title="Select Dictionary File",
            filetypes=[("Text files", "*.txt")]
        )
        if not dict_file:
            return
        
        try:
            best_key, decrypted = VigenereCipher.brute_force(
                ciphertext,
                dict_file,
                "brute_force_results.txt",
                alphabet=alphabet
            )

            output_path = self.output_entry.get()
            if output_path:
                self.write_to_file(output_path, decrypted)
            else:
                messagebox.showwarning('No Output file')
            
            self.res_text.delete("1.0", tk.END)
            self.res_text.insert(tk.END, decrypted)
            
            self.cipher_widgets['keyword_value'].delete(0, tk.END)
            self.cipher_widgets['keyword_value'].insert(0, best_key)
            
            messagebox.showinfo(
                "Brute Force Complete",
                f"Best key found: {best_key}\nResults saved to brute_force_results.txt"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Brute force failed: {str(e)}")

    @staticmethod
    def generate_matrix_for_affine_cipher(size, modulus, attempts=100):
        for i in range(attempts):
            matrix = np.random.randint(0, modulus, (size, size))
            
            try:
                det = int(round(np.linalg.det(matrix))) % modulus
                
                if det != 0 and math.gcd(det, modulus) == 1:
                    return matrix
            except:
                continue
            
    def generate_key_for_affine_cipher(self):
        matrix_size = int(self.cipher_widgets['matrix_size_entry'].get())
        alphabet_file = self.cipher_widgets['alphabet_entry'].get()
        alphabet = self.read_file(alphabet_file)
        modulus = len(alphabet)
        
        matrix_key = self.generate_matrix_for_affine_cipher(matrix_size, modulus)
        shift_vector = np.random.randint(0, modulus, matrix_size)

        self.cipher_widgets['matrix_entry'].delete(0, tk.END)
        self.cipher_widgets['matrix_entry'].insert(0, str(matrix_key.tolist()))
        
        self.cipher_widgets['vector_entry'].delete(0, tk.END)
        self.cipher_widgets['vector_entry'].insert(0, str(shift_vector.tolist()))


    def import_key_for_affine(self):
        file_path = filedialog.askopenfilename(
            title="Select Key File", 
            filetypes=[("Text files", "*.txt")]
        )
    
        with open(file_path, 'r') as f:
            content = f.readlines()

        matrix = None
        vector = None
        for line in content:
            line = line.strip()
            if line.startswith("Matrix:"):
                matrix_str = line.split("Matrix: ")[1]
                matrix = eval(matrix_str)
            elif line.startswith("Vector:"):
                vector_str = line.split("Vector: ")[1]
                vector = eval(vector_str)
    
        if matrix is None or vector is None:
            raise ValueError("Invalid key file format")
        
        matrix_size = len(matrix)

        self.cipher_widgets['matrix_size_entry'].delete(0, tk.END)
        self.cipher_widgets['matrix_size_entry'].insert(0, str(matrix_size))

        self.cipher_widgets['matrix_entry'].delete(0, tk.END)
        self.cipher_widgets['matrix_entry'].insert(0, str(matrix))
        
        self.cipher_widgets['vector_entry'].delete(0, tk.END)
        self.cipher_widgets['vector_entry'].insert(0, str(vector))
        
        matrix_size = len(matrix)
        self.cipher_widgets['matrix_size_entry'].delete(0, tk.END)
        self.cipher_widgets['matrix_size_entry'].insert(0, str(matrix_size))

    def export_key_for_affine(self):
       file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt")]
        )
       
       matrix_size = eval(self.cipher_widgets['matrix_size_entry'].get())
       matrix = eval(self.cipher_widgets['matrix_entry'].get())
       vector = eval(self.cipher_widgets['vector_entry'].get())

       with open(file_path, 'w') as f:
            f.write(f"Matrix size: {matrix_size}\n")
            f.write(f"Matrix: {matrix}\n")
            f.write(f"Vector: {vector}\n")
    
    def search_file(self, entry):
        file_name = filedialog.askopenfilename()
        entry.delete(0, tk.END)
        entry.insert(0, file_name)
    
    def process(self):
        tracemalloc.start()
        start_time = time.time()
        first_snapshot = tracemalloc.take_snapshot()

        input_file = self.input_entry.get()
        output_file = self.output_entry.get()
        cipher_name = self.selected_cipher.get()
        text = self.read_file(input_file)
        
        if not text:
            messagebox.showerror("Error", "Input file is empty or couldn't be read")
            return
        
        try:
            cipher_class = self.ciphers[cipher_name]
            cipher = self._create_cipher_instance(cipher_class)
            
            block_length = self._get_block_length()
            
            if self.encrypt_var.get():
                result = cipher_class.process_block(
                    text, 
                    block_length, 
                    encrypt=True, 
                    cipher=cipher
                )
            else:
                result = cipher_class.process_block(
                    text, 
                    block_length, 
                    encrypt=False, 
                    cipher=cipher
                )
                
            if isinstance(result, str):
                self.write_to_file(output_file, result)
                self.res_text.delete("1.0", tk.END)
                self.res_text.insert(tk.END, result)
            else:
                self.res_text.delete("1.0", tk.END)
                self.res_text.insert(tk.END, f"Error: Invalid result type {type(result)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {str(e)}")
            
        second_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        end_time = time.time()

        total_time = end_time - start_time
        snapshots = second_snapshot.compare_to(first_snapshot, 'lineno')
        total_memory = sum(stat.size_diff for stat in snapshots)
        
        self.time_label.config(text=f"Time taken: {total_time:.2f} seconds")
        self.memory_label.config(text=f"Memory usage: {total_memory / 1024:.2f} KiB")
    
    def _create_cipher_instance(self, cipher_class):
        cipher_name = self.selected_cipher.get()

        if cipher_name == "rail_fence":
            height = int(self.cipher_widgets['height_entry'].get())
            return cipher_class(height)
        
        elif cipher_name == "caesar":
            key = int(self.cipher_widgets['shear_value'].get())
            alphabet = self.read_file(self.cipher_widgets['alphabet_entry'].get())
            return cipher_class(key, alphabet)
            
        elif cipher_name == "cardano":
            size = int(self.cipher_widgets['key_size_entry'].get())
            return cipher_class(self.selected_cells, size)
        
        elif cipher_name == "vigenere":
            key = str(self.cipher_widgets['keyword_value'].get())
            alphabet = self.read_file(self.cipher_widgets['alphabet_entry'].get())
            return cipher_class(key, alphabet)
        
        elif cipher_name == "affine":
            matrix_size = int(self.cipher_widgets['matrix_size_entry'].get())
            matrix_str = self.cipher_widgets['matrix_entry'].get()
            vector_str = self.cipher_widgets['vector_entry'].get()
            alphabet = self.read_file(self.cipher_widgets['alphabet_entry'].get())

            if not alphabet:
                raise ValueError("Alphabet file is empty")
            
            key_matrix = np.array(eval(matrix_str), dtype=int)
            if key_matrix.shape != (matrix_size, matrix_size):
                raise ValueError(f"Key matrix must be {matrix_size}x{matrix_size}")
            
            shift_vector = np.array(eval(vector_str), dtype=int)
            if shift_vector.size != matrix_size:
                raise ValueError(f"Shift vector must be of length {matrix_size}")
            
            return AffineCipher(key_matrix, shift_vector, alphabet)
        
    
    def read_file(self, file_name):
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""
        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {str(e)}")
            return ""
        except FileNotFoundError:
            return ""
    
    def write_to_file(self, file_name, text):
        try:
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(text)
        except Exception as e:
            messagebox.showerror("Error", f"Error writing to file: {str(e)}")
