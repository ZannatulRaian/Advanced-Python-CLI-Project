import os
from project import analyze_text_file, list_files, save_report

DATA_DIR = "data"
TEST_FILE = os.path.join(DATA_DIR, "test.txt")

def test_analyze_text_file():
    # Create a test file
    with open(TEST_FILE, "w") as f:
        f.write("Hello world\nHello CS50\n")
    result = analyze_text_file(TEST_FILE)
    assert result["lines"] == 2
    assert result["words"] == 4
    assert result["unique_words"] == 3
    os.remove(TEST_FILE)

def test_list_files():
    # Create dummy files
    filenames = ["file1.txt", "file2.txt"]
    for name in filenames:
        open(os.path.join(DATA_DIR, name), "w").close()
    files = list_files(DATA_DIR)
    for name in filenames:
        assert name in files
        os.remove(os.path.join(DATA_DIR, name))

def test_save_report():
    report = {"test": 123}
    save_report(report, "dummy_report.json")
    assert os.path.exists(os.path.join("reports", "dummy_report.json"))
    os.remove(os.path.join("reports", "dummy_report.json"))
