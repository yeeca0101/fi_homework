import requests
from bs4 import BeautifulSoup

def get_news(stock_name):
    url = f"https://wts-info-api.tossinvest.com/api/v2/stock-infos?codes={stock_name}"
    url = f'https://wts-info-api.tossinvest.com/api/v2/stock-prices?codes={stock_name}'
    # url = f'https://wts-info-api.tossinvest.com/api/v2/stock-news?codes={stock_name}'
    url = 'https://wts-info-api.tossinvest.com/api/v2/forum/news/headline'
    
    response = requests.get(url)

    if response.status_code == 200:
        try:
            print(response.json())
        except:
            print(response.text)

        soup = BeautifulSoup(response.text, "html.parser")
        news_rows = soup.select("table.type5 tr")  # 뉴스 목록 행 선택
        news_data = []
        
        for row in news_rows:
            title_tag = row.select_one("td.title a")
            date_tag = row.select_one("td.date")
            
            if title_tag and date_tag:
                title = title_tag.get_text().strip()
                link = "https://finance.naver.com" + title_tag['href']
                date = date_tag.get_text().strip()
                news_data.append({"title": title, "link": link, "date": date})
        
        return news_data
    else:
        print("Failed to retrieve news information.")
        return None

# 예제 사용
stock_name = "US19990122001"  # 
news_data = get_news(stock_name)
print(news_data)
