静态页面抓取
python scraper.py static --urls https://example.com/article1 https://example.com/article2 --rate 1 --ttl 7200

分页 API
python scraper.py api --base-url "https://api.example.com/items?page={page}" --pages 10 --concurrency 4 --delay 0.3

动态渲染页面 (等待某元素并执行滚动加载)
python scraper.py dynamic --urls https://example.com/infinite --wait ".list-item" --scroll

Sitemap 增量抓取
python scraper.py sitemap --sitemap https://example.com/sitemap.xml --limit 50

三、依赖安装
pip install requests beautifulsoup4 lxml httpx[http2] playwright feedparser tqdm tenacity
playwright install chromium
