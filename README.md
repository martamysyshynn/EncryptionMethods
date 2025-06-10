# EncryptionMethods
This application provides a simple and user-friendly interface for encrypting and decrypting text using multiple encryption algorithms. It is ideal for learning, testing, or applying basic cryptographic methods. This program was created while studying the course “Fundamentals of Cryptography” at the Ivan Franko National University of Lviv.

# Supported Ciphers
1. Rail Fence Cipher
   - A transposition cipher that writes plaintext in a zigzag pattern across multiple "rails".
2. Caesar (Shift) Cipher with Frequency Analysis
   - Classic substitution cipher with support for frequency-based cryptanalysis to recover unknown keys.
3. Cardano Grille Cipher
   - A grid-based transposition cipher that uses a rotating key matrix.
   - Supports random key generation.
   - Allows key import/export from/to files.
4. Vigenère Cipher
   - A polyalphabetic substitution cipher using a keyword.
   - Implements Kasiski examination for key length estimation.
   - Supports brute-force key cracking using a dictionary (english_words.txt).
5. Higher-Order Affine Cipher
   - An extended version of the affine cipher allowing more complex substitutions.
   - Supports key generation.
   - Allows key import/export from/to files.



# File Structure

main.py - program launch
src:
   ciphers.py - algorithms for implementing ciphers
data:
   english_words.txt - dictionary
   english_alphabet.txt
   ukrainian_alphabet.txt
test:
   test_main.py - unit tests for the program
README.md – зroject documentation


# License

This project is for educational purposes only.
