import requests
from bs4 import BeautifulSoup
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# ==============================
# 1) Hàm tải HTML an toàn
# ==============================
def fetch(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        return None
    return None

# ==============================
# 2) Extract text từ HTML
# ==============================
def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Xoá script & style
    for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
        tag.extract()

    # Lấy toàn bộ text clean
    text = soup.get_text(separator="\n")
    # Loại bỏ dòng quá ngắn
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    return "\n".join(lines)

# ==============================
# 3) Các trang để crawl
# ==============================
URLS = [
    # báo
    "https://vnexpress.net/",
    "https://tuoitre.vn/",
    "https://dantri.com.vn/",
    "https://news.zing.vn/",
    # blog & review
    "https://www.tinhte.vn/",
    "https://vietcetera.com/vn",
    "https://genk.vn/",
    "https://cafef.vn/",
    # thêm vài trang tiếng Việt khác
    "https://kenh14.vn/",
    "https://cafebiz.vn/",
]

# ==============================
# 4) Crawl link từ trang chủ
# ==============================
def extract_links(home_html, home_url):
    soup = BeautifulSoup(home_html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Bỏ các link rác
        if href.startswith("javascript") or href.startswith("#"):
            continue

        # Chuyển relative → absolute
        if href.startswith("/"):
            href = home_url.rstrip("/") + href

        if href.startswith("http") and home_url.split("//")[1].split("/")[0] in href:
            links.append(href)

    # Lấy tối đa 40 link / trang
    return list(set(links))[:40]

# ==============================
# 5) Main
# ==============================
def crawl_all(output="tiny.txt"):
    all_text = []

    for url in URLS:
        print(f"Crawl trang: {url}")

        home = fetch(url)
        if not home:
            print("  ❌ Không tải được trang")
            continue

        links = extract_links(home, url)
        print(f"  Tìm được {len(links)} link bài")

        for link in links:
            print(f"    -> {link}")
            html = fetch(link)
            if not html:
                continue

            text = extract_text(html)
            if len(text) < 200:
                continue

            all_text.append(text)

            # nghỉ random để tránh bị chặn
            time.sleep(random.uniform(0.5, 1.5))

    print(f"\n📌 Tổng số bài lấy được: {len(all_text)}")
    joined = "\n\n".join(all_text)

    with open(output, "w", encoding="utf-8") as f:
        f.write(joined)

    print(f"✅ Ghi xong vào file: {output}")


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    crawl_all("tiny.txt")
