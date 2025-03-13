import time
import tracemalloc
import tkinter as tk
from tkinter import filedialog
import rail_fence_cipher
import caesar_cipher


def search_file(entry):
    file_name = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_name)

def process(method):
    tracemalloc.start()
    start_time = time.time()
    first_snapshot = tracemalloc.take_snapshot()

    input_file = input_entry.get()
    output_file = output_entry.get()

    text = read_file(input_file)

    if method == 'rail_fence_encrypt':
        height = int(height_entry.get())
        block_length = int(block_length_entry.get())
        res = rail_fence_cipher.fence_cipher_encrypt_by_blocks(text, height, block_length)
    elif method == 'rail_fence_decrypt':
        height = int(height_entry.get())
        block_length = int(block_length_entry.get())
        res = rail_fence_cipher.fence_cipher_decrypt_by_blocks(text, height, block_length)
    elif method == 'cesar_cipher_encrypt':
        alphabet = read_file(alphabet_entry.get())
        key = int(shear_value.get())
        block_length = int(block_length_entry.get())
        res = caesar_cipher.caesars_cipher_encrypt_by_blocks(text, block_length, alphabet, key)
    elif method == 'cesar_cipher_decrypt':
        alphabet = read_file(alphabet_entry.get())
        key = int(shear_value.get())
        block_length = int(block_length_entry.get())
        res = caesar_cipher.caesars_cipher_decrypt_by_blocks(text, block_length, alphabet, key)
    elif method == 'freq_analysis':
        alphabet = read_file(alphabet_entry.get())
        file_name = "frequency_analysis.txt"  
        key = caesar_cipher.frequency_analysis(text, alphabet, file_name)
        if key is None:
            res = "Frequency analysis failed. Key could not be determined."
            key_label.config(text="Determined key: N/A")
        else:
            res = caesar_cipher.caesars_cipher_decrypt_by_frequency_analysis(text, alphabet, file_name)
            key_label.config(text=f"Determined key: {key}")
    
    write_to_file(output_file, res)
    res_text.delete("1.0", tk.END)
    res_text.insert(tk.END, res)

    second_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    end_time = time.time()

    total_time = end_time - start_time
    snapshots = second_snapshot.compare_to(first_snapshot, 'lineno')
    total_memory = sum(stat.size_diff for stat in snapshots)
    
    time_label.config(text=f"Time taken: {total_time:.2f} seconds")
    memory_label.config(text=f"Memory usage: {total_memory / 1024:.2f} KiB")

def read_file(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        return ""

def write_to_file(file_name, text):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)

root = tk.Tk()
root.title('Encryption & Decryption')
root.geometry('850x500')

tk.Label(root, text='Input file: ').grid(row=0, column=0, padx=5, pady=5)
input_entry = tk.Entry(root, width=50)
input_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text='Browse', command=lambda: search_file(input_entry)).grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text='Output file: ').grid(row=1, column=0, padx=5, pady=5)
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text='Browse', command=lambda: search_file(output_entry)).grid(row=1, column=2, padx=5, pady=5)

tk.Label(root, text='Alphabet file: ').grid(row=2, column=0)
alphabet_entry = tk.Entry(root, width=50)
alphabet_entry.grid(row=2, column=1)
tk.Button(root, text='Browse', command=lambda: search_file(alphabet_entry)).grid(row=2, column=2)

tk.Label(root, text='Height (Rail Fence Cipher): ').grid(row=3, column=0)
height_entry = tk.Entry(root, width=10)
height_entry.grid(row=3, column=1, sticky='w')

tk.Label(root, text='Shear value (Ceasar Cipher): ').grid(row=4, column=0)
shear_value = tk.Entry(root, width=10)
shear_value.grid(row=4, column=1, sticky='w')

tk.Label(root, text='Block length: ').grid(row=5, column=0)
block_length_entry = tk.Entry(root, width=10)
block_length_entry.grid(row=5, column=1, sticky='w')

tk.Button(root, text='Rail Fence Encrypt', command=lambda: process('rail_fence_encrypt')).grid(row=6, column=0)
tk.Button(root, text='Rail Fence Decrypt', command=lambda: process('rail_fence_decrypt')).grid(row=6, column=1)
tk.Button(root, text='Caesar Encrypt', command=lambda: process('cesar_cipher_encrypt')).grid(row=7, column=0)
tk.Button(root, text='Caesar Decrypt', command=lambda: process('cesar_cipher_decrypt')).grid(row=7, column=1)
tk.Button(root, text='Frequency Analysis', command=lambda: process('freq_analysis')).grid(row=7, column=2)

res_text = tk.Text(root, height=10, width=70)
res_text.grid(row=8, column=0, columnspan=3)

key_label = tk.Label(root, text="Determined key: ")
key_label.grid(row=9, column=0, columnspan=3)

time_label = tk.Label(root, text="Time taken: ")
time_label.grid(row=10, column=0, columnspan=3)

memory_label = tk.Label(root, text="Memory usage: ")
memory_label.grid(row=11, column=0, columnspan=3)



root.mainloop()