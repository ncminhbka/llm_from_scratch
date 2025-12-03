import re
from bs4 import BeautifulSoup

def preprocess_file_remove_links(input_path="tiny.txt", output_path="clean_tiny.txt", min_line_len=40):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Loại bỏ HTML
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator="\n")

    # Loại bỏ URL (http, https, www)
    text = re.sub(r'https?://\S+', '', text)  # loại bỏ http:// hoặc https://
    text = re.sub(r'www\.\S+', '', text)      # loại bỏ www.example.com

    # Loại bỏ dòng quá ngắn và khoảng trắng thừa
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) >= min_line_len]

    # Gom lại
    clean_text = "\n".join(lines)

    # Lưu file mới
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"✅ Tiền xử lý xong, lưu vào {output_path}")
    print(f"Tổng số dòng sau xử lý: {len(lines)}")

preprocess_file_remove_links("tiny.txt", "clean_tiny.txt")
