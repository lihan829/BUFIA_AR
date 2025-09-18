# BUFIA-AR

**BUFIA-AR (Bottom-Up Factor Inference Algorithm over Autosegmental Representations)** is a tool for learning phonological grammars from syllabified wordlists.  
It represents words as graph-structured inputs (tones, syllables, moras) and induces constraints/grammars over them.

---

## ✨ Features
- Convert syllabified or **orthographic** wordlists into **Autorep** graph objects  
- Print out the corresponding **autosegmental representations** of words  
- Infer phonological grammars (inviolable constraints over ARs) with user-defined complexity limits  
- Export learned constraints as `svg`, `png`, or `pdf` — ready to include directly in papers

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/BUFIA_AR.git
cd BUFIA_AR
```

### 2. Create the environment
An `environment.yml` file is provided with all required dependencies.  
Create the environment with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate bufia-ar
```

### 3. Verify installation
```bash
python -c "import graphviz, numpy; print(graphviz.__version__, numpy.__version__)"
dot -V   # checks Graphviz binary
```

---

## 🚀 Usage

### Basic syntax
```bash
python BUFIA_AR.py --input INPUT [--output OUTPUT] [--format FORMAT]
                   [--t T] [--s S] [--m M] (-f | --learn)
```

### Required arguments
- `--input INPUT` : path to syllabified or orthographic wordlist  

Choose one mode:
- `-f, --file` : print the Autolist (all Autorep objects)  
- `--learn` : learn grammar rules from the input  

### Optional arguments
- `--output OUTPUT` : output directory (default: `output`)  
- `--format FORMAT` : graph output format (`svg`, `png`, `pdf`; default: `svg`)  
- `--t T` : tone number limit (default: `2`)  
- `--s S` : syllable number limit (default: `2`)  
- `--m M` : mora number limit (default: `2`)  

---

## 📚 Examples

**1. Print Autolist**

The list of tuples `(tone, mora, syllable)` encodes associations across three tiers.  
For example, given:

```python
[(1, 1, 1), (1, 2, 2), (1, 3, 2)]
```

This means:
- `(1, 1, 1)` → tone 1 connects to mora 1 and syllable 1  
- `(1, 2, 2)` → tone 1 connects to mora 2 and syllable 2  
- `(1, 3, 2)` → tone 1 connects to mora 3 and syllable 2  

To print the Autolist for a wordlist:

```bash
python BUFIA_AR.py --input data/wordlist.txt -f
```

---

**2. Learn grammar with default settings**

The default configuration learns grammars within **2 tones, 2 moras, and 2 syllables**.  
The output will be saved as `.svg` files in the `output/` directory.

```bash
python BUFIA_AR.py --input data/hausa.txt --learn
```

---

**3. Learn grammar with custom settings**

You can increase the complexity limits and change the output format.  
For example, to allow up to **3 tones, 3 moras, and 3 syllables**, save results in `results/`, and export plots as PNG:

```bash
python BUFIA_AR.py --input data/hausa.txt --learn \
                   --output results --format png --t 3 --s 3 --m 3
```

---

## 📂 Project Structure
```
BUFIA_AR/
│
├── BUFIA_AR.py        # main entry script
├── autorep.py         # Autorep class for graph representations
├── data/              # sample syllabified wordlists
└── output/            # generated graphs and grammars
```

---

## 📖 Related Works

- Chandlee, J., Eyraud, R., Heinz, J., Jardine, A., & Rawski, J. (2019, July). Learning with Partially Ordered Representations. In *Proceedings of the 16th Meeting on the Mathematics of Language* (pp. 91-101).


- Li, H. (2025).  *Learning Tonotactic Patterns over Autosegmental Representations.*  In *Proceedings of the Annual Meetings on Phonology* (Vol. 1, No. 1). University of Massachusetts Amherst Libraries.

