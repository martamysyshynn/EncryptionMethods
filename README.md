# EncryptionMethods
This application provides a simple and user-friendly interface for encrypting and decrypting text using multiple encryption algorithms. It is ideal for learning, testing, or applying basic cryptographic methods. This program was created while studying the course “Fundamentals of Cryptography” at the Ivan Franko National University of Lviv.

# Supported Ciphers

1. Rail Fence Cipher
   - A transposition cipher that writes plaintext in a zigzag pattern across multiple "rails".
   - Supports customizable rail height and block processing.
2. Caesar (Shift) Cipher
   - Classic substitution cipher with support for frequency-based cryptanalysis to recover unknown keys.
   - Supports custom alphabets and block-wise encryption/decryption.
3. Cardano Grille Cipher
   - A grid-based transposition cipher that uses a rotating key matrix.
   - Features:
        - Random key generation.
        - Key import/export from/to files.
        - Interactive grid for key configuration.
4. Vigenere Cipher
   - A polyalphabetic substitution cipher using a keyword.
   - Advanced features:
      - Kasiski examination for key length estimation.
      - Brute-force key cracking using a dictionary (e.g., english_words.txt).
5. Higher-Order Affine Cipher
   - An extended version of the affine cipher allowing complex matrix-based substitutions.
   - Features:
      - Random key generation.
      - Key import/export from/to files.
      - Support for block-wise processing.


# File Structure

- main.py - program launch
- src:
   - ciphers.py - algorithms for implementing ciphers
- data:
   - english_words.txt - dictionary
   - english_alphabet.txt
   - ukrainian_alphabet.txt
- test:
   - test_main.py - unit tests for the program
- README.md – project documentation

# Installation and Usage
   # Prerequisites
      - Python 3.x
      - Tkinter 
      - Libraries: tracemalloc, collections, random, math, functools, string, numpy, unittest, os

   # Steps to Run

      1. Clone the repository or download the source files.
      2. Install required libraries (if not already installed).
      3. Run the application.

# Additional Notes

- Performance Metrics
   - The application measures time taken and memory usage for each operation, displayed in the GUI.
- Key Management
   - Cardano and Affine Ciphers: Keys can be exported/imported for reuse.
   - Dictionary Attacks: For Vigenere, a dictionary file (english_words.txt) is required for brute-force attacks.
- Limitations
   - Non-alphabetic characters (e.g., numbers, symbols) may not be processed by some ciphers.
   - Large files may impact performance due to memory constraints.

# License

This project is for educational purposes only.
