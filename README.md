# Python CLI Project

![Python](https://img.shields.io/badge/python-3.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-yellow)

---

## Project Structure

```
project/
│
├─ data/                  # Input folder for text and CSV files
├─ reports/               # Output folder for JSON analysis reports
├─ project.py             # Main Python program with CLI commands
├─ test_project.py        # Unit tests for custom functions
├─ requirements.txt       # List of pip dependencies
└─ README.md              # Project documentation
```

---

## Project Overview

This is an **Python CLI tool** that demonstrates a broad set of programming concepts suitable for learning and real-world applications. The project includes:

- **OS interaction & File I/O**: Read, write, and manage files and directories.
- **Generators & Iterators**: Handle large datasets efficiently.
- **Data Structures**: Lists, sets, dictionaries, and comprehensions.
- **JSON Reports**: Save structured data analysis results.
- **Command-Line Interface (CLI)**: Run specific functions directly using `argparse`.
- **Data Analysis**: Use `pandas` and `numpy` for numerical computations.
- **Machine Learning**: Train simple models using `scikit-learn`.
- **Data Visualization**: Plot graphs with `matplotlib`.
- **Professional Structure**: Main function, modular functions, and testable design.

---

## Installation

1. **Clone the repository:**

```bash
git clone <repo-url>
cd <repo-folder>
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
venv\\Scripts\\activate       # Windows
# OR
source venv/bin/activate    # macOS / Linux
```

3. **Install required packages:**

```bash
pip install -r requirements.txt
```

---

## Usage

Run the program using the **command-line interface**.

### 1. List files in the data/ folder

```bash
python project.py list-files
```

- Output example:

```
Files in data folder: ['myfile.txt', 'data.csv']
```

---

### 2. Analyze a text file

```bash
python project.py analyze-text myfile.txt
```

- Generates a **JSON report** in `reports/` with:
  - Number of lines
  - Number of words
  - Number of unique words
- Example report saved as `reports/myfile_report.json`:

```json
{
  "file": "myfile.txt",
  "lines": 42,
  "words": 350,
  "unique_words": 210
}
```

---

### 3. Analyze a CSV file

```bash
python project.py analyze-csv data.csv
```

- Shows statistical summary of numeric columns using `pandas`.
- Example output:

```
       Age     Salary
count  100.0    100.0
mean    35.5   56000.0
std      5.2    8000.0
min     22.0   40000.0
max     48.0   75000.0
```

---

### 4. Train a Machine Learning model

```bash
python project.py train-ml data.csv target_column
```

- Uses **Linear Regression** to predict the target column.
- Outputs **Mean Squared Error (MSE)**.
- Example output:

```
Training Linear Regression model on target column: Salary
Mean Squared Error: 6250000.0
```

---

### 5. Plot a column from a CSV

```bash
python project.py plot data.csv column_name
```

- Opens a line chart of the selected column using `matplotlib`.
- Supports multiple columns in future expansions.

---

## Testing

Unit tests are implemented using **pytest**.  
To run all tests:

```bash
pytest test_project.py
```

- Each function is independently tested for correctness.

---

## Dependencies

- `pandas` – data manipulation  
- `numpy` – numerical computation  
- `scikit-learn` – machine learning  
- `matplotlib` – data visualization  

All dependencies are listed in `requirements.txt`.

---

## Example CLI Commands

```bash
# List files in data folder
python project.py list-files

# Analyze text file
python project.py analyze-text myfile.txt

# Analyze CSV data
python project.py analyze-csv data.csv

# Train ML model
python project.py train-ml data.csv Salary

# Plot column
python project.py plot data.csv Salary
```

---

## Notes

- Place all input files (text or CSV) in the `data/` folder.
- Reports are automatically saved in the `reports/` folder.
- For large datasets, the program uses **generators** to avoid memory issues.
- Extend the CLI by adding new commands or functions to `project.py`.

---

## License

This project is licensed under the **MIT License**.  

---
