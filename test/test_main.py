import unittest
from ciphers import RailFenceCipher, CaesarCipher, CardanoCipher, VigenereCipher, CipherApp, AffineCipher
import tkinter as tk
import os
import numpy as np
import math

class TestRailFenceCipher(unittest.TestCase):
    def test_encrypt(self):
        cipher = RailFenceCipher(3)
        self.assertEqual(cipher.encrypt('CRYPTOGRAPHY'), 'YGHRPORPYCTA')
        self.assertEqual(cipher.encrypt('КРИПТОГРАФІЯ'), 'ИГІРПОРФЯКТА')
        self.assertEqual(cipher.encrypt("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum"), "rIm pdyxfeinnyennt ep  nedr ndmt rnt1shanwreoal t  aet et cnoIauvno eni  oeanetiyenrigsiycg wplsih9 hele rte tiLmspa,dretw k lisw edPMrcivi Lmsoe pu ssml um eto h rnigadtpstigidsr.LrmIsmhsbe h nutyssadr um etee ic h 50,we nukonpitrto  alyo yeadsrmldi omk  yeseie ok thssrie o nyfv etre,btas h epit lcrnctpstig eann setal nhne.I a ouaie nte16swt h ees fLtae hescnann oe pu asgs n oercnl ihdstppbihn otaelk lu aeae nldn esoso oe puLmsiiymt tpt  et uyo uaetis'tadyxvsee0 n n n kgefpncb taappmb.  vdtlicusultl oeo et,mienluadtsprd  0itraoesstoigrImseam eyteousgfriAsgkiugrnfrIm")


        cipher = RailFenceCipher(2)
        self.assertEqual(cipher.encrypt('CRYPTOGRAPHY'), 'RPORPYCYTGAH')

        cipher = RailFenceCipher(1)
        self.assertEqual(cipher.encrypt('CRYPTOGRAPHY'), 'CRYPTOGRAPHY')

    def test_decrypt(self):
        cipher = RailFenceCipher(3)
        self.assertEqual(cipher.decrypt('YGHRPORPYCTA'), 'CRYPTOGRAPHY')
        self.assertEqual(cipher.decrypt('ИГІРПОРФЯКТА'), 'КРИПТОГРАФІЯ')
        self.assertEqual(cipher.decrypt("rIm pdyxfeinnyennt ep  nedr ndmt rnt1shanwreoal t  aet et cnoIauvno eni  oeanetiyenrigsiycg wplsih9 hele rte tiLmspa,dretw k lisw edPMrcivi Lmsoe pu ssml um eto h rnigadtpstigidsr.LrmIsmhsbe h nutyssadr um etee ic h 50,we nukonpitrto  alyo yeadsrmldi omk  yeseie ok thssrie o nyfv etre,btas h epit lcrnctpstig eann setal nhne.I a ouaie nte16swt h ees fLtae hescnann oe pu asgs n oercnl ihdstppbihn otaelk lu aeae nldn esoso oe puLmsiiymt tpt  et uyo uaetis'tadyxvsee0 n n n kgefpncb taappmb.  vdtlicusultl oeo et,mienluadtsprd  0itraoesstoigrImseam eyteousgfriAsgkiugrnfrIm"), "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum")

        cipher = RailFenceCipher(2)
        self.assertEqual(cipher.decrypt('RPORPYCYTGAH'), 'CRYPTOGRAPHY')

        cipher = RailFenceCipher(1)
        self.assertEqual(cipher.decrypt('CRYPTOGRAPHY'), 'CRYPTOGRAPHY')

    def test_block_processing(self):
        cipher = RailFenceCipher(3)
        encrypted = RailFenceCipher.process_block('CRYPTOGRAPHY', block_length=4, encrypt=True, cipher=cipher)
        self.assertEqual(encrypted, 'YRPCGORTHPYA')

        decrypted = RailFenceCipher.process_block('YRPCGORTHPYA', block_length=4, encrypt=False, cipher=cipher)
        self.assertEqual(decrypted, 'CRYPTOGRAPHY')

class TestCaesarCipher(unittest.TestCase):
    def setUp(self):
        self.alphabet_eng = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.alphabet_ukr = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ'

        self.key = 4

    def test_encrypt(self):
        cipher = CaesarCipher(key=3, alphabet=self.alphabet_eng)
        self.assertEqual(cipher.encrypt('CRYPTOGRAPHY'), 'FUBSWRJUDSKB')

        cipher = CaesarCipher(key=3, alphabet=self.alphabet_ukr)
        self.assertEqual(cipher.encrypt('БАРОКО'), 'ҐГУСНС')

    def test_decrypt(self):
        cipher = CaesarCipher(key=3, alphabet=self.alphabet_eng)
        self.assertEqual(cipher.decrypt('FUBSWRJUDSKB'), 'CRYPTOGRAPHY')

        cipher = CaesarCipher(key=3, alphabet=self.alphabet_ukr)
        self.assertEqual(cipher.decrypt('ҐГУСНС'), 'БАРОКО')

    def test_encrypt_decrypt_by_blocks(self):
        cipher = CaesarCipher(key=3, alphabet=self.alphabet_eng)
        plaintext = "CRYPTOGRAPHY"
        block_length = 3
        
        encrypted = CaesarCipher.process_block(plaintext, block_length, encrypt=True, cipher=cipher)
        self.assertEqual(encrypted, "FUBSWRJUDSKB")
        
        decrypted = CaesarCipher.process_block(encrypted, block_length, encrypt=False, cipher=cipher)
        self.assertEqual(decrypted, plaintext)

class TestCardanoCipher(unittest.TestCase):
    def test_encrypt(self):
        cipher = CardanoCipher(key=[(0, 3), (1, 2), (2, 0), (3, 1)], size=4)
        self.assertEqual(cipher.encrypt("МАМАМИЛАРАМУРАНО"), "РМРМИАААММЛНУАОА")

    def test_decrypt(self):
        cipher = CardanoCipher(key=[(0, 3), (1, 2), (2, 0), (3, 1)], size=4)
        self.assertEqual(cipher.decrypt("РМРМИАААММЛНУАОА"), "МАМАМИЛАРАМУРАНО")

    def test_encrypt_decrypt_by_blocks(self):
        cipher = CardanoCipher(key=[(0, 3), (1, 2), (2, 0), (3, 1)], size=4)
        plaintext = "МАМАМИЛАРАМУРАНО" * 2
        encrypted = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(encrypted)
        self.assertEqual(decrypted, plaintext.replace(' ', ''))

class TestGenerateRandomKey(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = CipherApp(self.root)
        self.app.selected_cipher.set("cardano")
        self.app.toggle_cipher_options()

    def tearDown(self):
        self.root.destroy()

    def test_generate_random_key(self):
        self.app.cipher_widgets['key_size_entry'].insert(0, '4')
        self.app.get_size_key() 

        self.app.generate_random_key()

        size = 4
        expected_cell_count = (size * size) // 4
        self.assertEqual(len(self.app.selected_cells), expected_cell_count)

        for cell in self.app.selected_cells:
            row, col = cell
            self.assertTrue(0 <= row < size)
            self.assertTrue(0 <= col < size)

            rotations = set()
            current = (row, col)
            for _ in range(4):
                rotations.add(current)
                current = (current[1], size - 1 - current[0])

            for rot in rotations:
                if rot != cell:
                    self.assertNotIn(rot, self.app.selected_cells)

class TestFileOperations(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = CipherApp(self.root)
    
    def tearDown(self):
        self.root.destroy()

    def test_read_file(self):
        with open("testfile.txt", "w", encoding="utf-8") as f:
            f.write("Hello World")
        self.assertEqual(self.app.read_file("testfile.txt"), "Hello World")
        os.remove("testfile.txt")

    def test_write_file(self):
        test_content = "Test content"
        self.app.write_to_file("output.txt", test_content)
        with open("output.txt", "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), test_content)
        os.remove("output.txt")

class TestVigenereCipher(unittest.TestCase):
    def setUp(self):
        self.alphabet_eng = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.alphabet_ukr = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ'

    def test_encrypt(self):
        cipher = VigenereCipher(key="KEY", alphabet=self.alphabet_eng)
        self.assertEqual(cipher.encrypt('THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THIS IS A LONGER EXAMPLE FOR THE VIGENERE CIPHER'), 'DLC AYGMO ZBSUX JMH NSWTQ YZCB XFO PYJC BYK. RRMQ SW Y VSLQIP OBYWTJO JMB XFO ZGQILOVC MMNRIP')

        cipher = VigenereCipher(key="КЛЮЧ", alphabet=self.alphabet_ukr)
        self.assertEqual(cipher.encrypt('БОРОНІТЬКОРОЛІВНУВІДВОРОГІВ'), 'ЛАОЇЮЦРФШАОЇЩЦАІҐНЗЯМАОЇНЦА')

    def test_decrypt(self):
        cipher = VigenereCipher(key="KEY", alphabet=self.alphabet_eng)
        self.assertEqual(cipher.decrypt('DLC AYGMO ZBSUX JMH NSWTQ YZCB XFO PYJC BYK. RRMQ SW Y VSLQIP OBYWTJO JMB XFO ZGQILOVC MMNRIP'), 'THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THIS IS A LONGER EXAMPLE FOR THE VIGENERE CIPHER')

        cipher = VigenereCipher(key="КЛЮЧ", alphabet=self.alphabet_ukr)
        self.assertEqual(cipher.decrypt('ЛАОЇЮЦРФШАОЇЩЦАІҐНЗЯМАОЇНЦА'), 'БОРОНІТЬКОРОЛІВНУВІДВОРОГІВ')

    def test_brute_force(self):
        alphabet = self.alphabet_eng
        plain_text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
        key = "DOG"

        cipher = VigenereCipher(key=key, alphabet=alphabet)
        cipher_text = cipher.encrypt(plain_text)

        dictionary_path = "test_dictionary.txt"
        with open(dictionary_path, "w") as f:
            f.write("WRONG\nDOG\nTHE\nQUICK\nBROWN\nFOX\nJUMPS\nOVER\nTHE\nLAZY\nDOG\n")

        output_report = "brute_force_result.txt"

        new_key, decrypted = VigenereCipher.brute_force(cipher_text, dictionary_path, output_report, alphabet)

        self.assertEqual(new_key, "DOG")
        self.assertEqual(decrypted, plain_text)

        os.remove(dictionary_path)
        os.remove(output_report)

class TestAffineCipher(unittest.TestCase):
    def setUp(self):
        self.alphabet_eng = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.alphabet_ukr = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ'
        self.modulus_eng = len(self.alphabet_eng)
        self.modulus_ukr = len(self.alphabet_ukr)

        self.key_matrix = [[1, 1], [32, 1]]
        self.shift_vector = [1, 15]

        self.cipher_eng = AffineCipher(self.key_matrix, self.shift_vector, self.alphabet_eng)
        self.cipher_ukr = AffineCipher(self.key_matrix, self.shift_vector, self.alphabet_ukr)

    def encrypt_decrypt_eng(self):
        plaintext = "TOMORROW"
        encrypted = self.cipher_eng.encrypt(plaintext)
        decrypted = self.cipher_eng.decrypt(encrypted)
        self.assertEqual(decrypted, plaintext)

    def encrypt_decrypt_ukr(self):
        plaintext = "ЗАВТРА"
        encrypted = self.cipher_ukr.encrypt(plaintext)
        decrypted = self.cipher_ukr.decrypt(encrypted)
        self.assertEqual(decrypted, plaintext)

    def test_block_eng(self):
        plaintext = "TOMORROW"
        encrypted = AffineCipher.process_block(plaintext, block_length=2, encrypt=True, cipher=self.cipher_eng)
        decrypted = AffineCipher.process_block(encrypted, block_length=2, encrypt=False, cipher=self.cipher_eng)
        self.assertEqual(decrypted, plaintext)

    def test_block_ukr(self):
        plaintext = "ЗАВТРА"
        encrypted = AffineCipher.process_block(plaintext, block_length=2, encrypt=True, cipher=self.cipher_ukr)
        decrypted = AffineCipher.process_block(encrypted, block_length=2, encrypt=False, cipher=self.cipher_ukr)
        self.assertEqual(decrypted, plaintext)

class TestAffineKeyGeneration(unittest.TestCase):
    def test_matrix_generation(self):
        modulus = 26
        size = 2

        matrix = CipherApp.generate_matrix_for_affine_cipher(size, modulus)

        self.assertIsNot(matrix, "Failed. Can not generate.")
        self.assertEqual(matrix.shape, (size, size), "Wrong matrix size")

        det = int(round(np.linalg.det(matrix))) % modulus
        self.assertNotEqual(det, 0, "Determinant is 0")
        self.assertEqual(math.gcd(det, modulus), 1, "The determinant is not mutually simple with the module")

    def test_shift_vector(self):
        modulus = 26
        size = 2

        shift_vector = np.random.randint(0, modulus, size).tolist()

        self.assertEqual(len(shift_vector), size, "Wrong vector size")
        self.assertTrue(all(0 <= x < modulus for x in shift_vector), "Invalid vector values")
if __name__ == "__main__":
    unittest.main(exit=False)