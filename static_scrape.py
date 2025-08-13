# -*- coding: utf-8 -*-
"""
多场景合规数据采集脚本示例合集
说明：
  - 仅供学习/公开授权数据采集，请遵守目标站点 Robots.txt、服务条款、版权及法律法规
  - 不要采集登录后、付费、个人隐私、受保护或禁止抓取的数据
  - 合理设置速率，避免造成服务器压力；必要时联系站点获取正式 API
依赖建议：
  pip install requests beautifulsoup4 lxml httpx[http2] aiofiles playwright feedparser tqdm tenacity sqlite-utils
  playwright install chromium
结构：
  1. utils/：通用工具（缓存、限速、指纹、重试、日志）
  2. 示例脚本：
     a) static_scrape.py  —— 静态页面抓取 + 解析 + SQLite 存储
     b) api_crawler.py    —— 分页 JSON API 并发抓取（速率/重试/断点）
     c) dynamic_scrape.py —— 动态渲染页面（Playwright headless）
     d) sitemap_scrape.py —— Sitemap 抽取 URL + 增量更新
可按需拆成多个文件；此处为单文件演示。
"""

import os
import re
import sys
import json
import time
import math
import sqlite3
import asyncio
import logging
import hashlib
import random
import contextlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Iterable, Tuple, Callable

import requests
from bs4 import BeautifulSoup

try:
    import httpx
except ImportError:
    httpx = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    retry = None

# -------------------------------
# 通用配置 & 日志
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("scraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/125.0"
]

def pick_ua():
    return random.choice(USER_AGENTS)

# -------------------------------
# 轻量文件缓存 (GET)
# -------------------------------
class SimpleCache:
    def __init__(self, cache_dir="cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def get(self, url: str) -> Optional[Dict[str, Any]]:
        p = self._key(url)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > self.ttl:
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None

    def set(self, url: str, data: Dict[str, Any]):
        p = self._key(url)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

# -------------------------------
# 限速器
# -------------------------------
class RateLimiter:
    def __init__(self, rate: float):
        """
        rate: 每秒最大请求数 (e.g. 2 => 间隔 ~0.5 秒)
        """
        self.min_interval = 1.0 / rate if rate > 0 else 0
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()

# -------------------------------
# SQLite 简易封装
# -------------------------------
class ArticleStore:
    def __init__(self, db_path="data.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          url TEXT UNIQUE,
          title TEXT,
          author TEXT,
          published TEXT,
          summary TEXT,
          content TEXT,
          raw_html TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        self.conn.commit()

    def upsert(self, rec: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO articles (url,title,author,published,summary,content,raw_html)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
          title=excluded.title,
          author=excluded.author,
          published=excluded.published,
          summary=excluded.summary,
          content=excluded.content,
          raw_html=excluded.raw_html
        """, (
            rec.get("url"),
            rec.get("title"),
            rec.get("author"),
            rec.get("published"),
            rec.get("summary"),
            rec.get("content"),
            rec.get("raw_html"),
        ))
        self.conn.commit()

# ===========================================================
# a) 静态页面抓取示例 (requests + BeautifulSoup + 缓存 + 限速)
# ===========================================================
def static_scrape(seed_urls: List[str], cache_ttl=3600, rate=1.5):
    logger.info("静态抓取启动: %d 个种子", len(seed_urls))
    cache = SimpleCache(ttl=cache_ttl)
    limiter = RateLimiter(rate=rate)
    store = ArticleStore()

    headers_base = {
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    for url in seed_urls:
        cached = cache.get(url)
        if cached:
            logger.info("命中缓存: %s", url)
            html = cached["html"]
        else:
            limiter.wait()
            resp = requests.get(url, headers={"User-Agent": pick_ua(), **headers_base}, timeout=15)
            if resp.status_code != 200:
                logger.warning("跳过: %s 状态码=%s", url, resp.status_code)
                continue
            html = resp.text
            cache.set(url, {"html": html, "fetched_at": time.time()})

        soup = BeautifulSoup(html, "lxml")

        # 基础解析逻辑 (需针对目标站点定制)
        title = soup.select_one("h1") or soup.select_one("title")
        title_text = title.get_text(strip=True) if title else "N/A"
        author = ""
        meta_author = soup.select_one("meta[name=author]")
        if meta_author:
            author = meta_author.get("content", "")
        published = ""
        time_el = soup.find("time")
        if time_el:
            published = time_el.get("datetime") or time_el.get_text(strip=True)

        # 取正文 (示例：所有 <p>)
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p")] or \
                     [p.get_text(" ", strip=True) for p in soup.select("p")]
        content = "\n".join(paragraphs[:50])  # 限制长度防止爆炸
        summary = content[:160]

        store.upsert({
            "url": url,
            "title": title_text,
            "author": author,
            "published": published,
            "summary": summary,
            "content": content,
            "raw_html": html[:200000]  # 上限避免太大
        })
        logger.info("保存文章: %s | %s", title_text, url)

# ===========================================================
# b) JSON API 并发分页抓取 (httpx + asyncio)
# ===========================================================
async def api_crawler(base_url: str,
                      page_param="page",
                      start=1,
                      limit_pages=5,
                      concurrency=5,
                      delay=0.2,
                      outfile="api_data.json"):
    """
    base_url: 如 https://api.example.com/items?page={page}
    注意：此处仅示意公开/许可 API；请勿绕过鉴权或抓取受保护接口
    """
    if httpx is None:
        raise RuntimeError("需要安装 httpx: pip install httpx[http2]")

    sem = asyncio.Semaphore(concurrency)
    session_headers = {
        "User-Agent": pick_ua(),
        "Accept": "application/json",
    }

    async def fetch(client: httpx.AsyncClient, page: int):
        url = base_url.format(page=page)
        async with sem:
            await asyncio.sleep(delay + random.random()*0.1)
            try:
                r = await client.get(url, timeout=15)
                if r.status_code != 200:
                    logger.warning("页面 %s 状态=%s", page, r.status_code)
                    return None
                return r.json()
            except Exception as e:
                logger.error("请求失败 page=%s err=%s", page, e)
                return None

    collected: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(http2=True, headers=session_headers) as client:
        tasks = []
        for p in range(start, start + limit_pages):
            tasks.append(asyncio.create_task(fetch(client, p)))
        for coro in asyncio.as_completed(tasks):
            data = await coro
            if not data:
                continue
            # 假设数据在 data["items"]
            items = data.get("items") or data.get("data") or []
            for item in items:
                # 清理/规范子集
                cleaned = {
                    "id": item.get("id"),
                    "title": item.get("title") or item.get("name"),
                    "raw": item
                }
                collected.append(cleaned)

    Path(outfile).write_text(json.dumps(collected, ensure_ascii=False, indent=2), "utf-8")
    logger.info("API 抓取完成, %d 条记录写入 %s", len(collected), outfile)

# ===========================================================
# c) 动态渲染抓取 (Playwright)
# ===========================================================
async def dynamic_scrape(urls: List[str],
                         wait_selector: Optional[str] = None,
                         scroll: bool = False,
                         headless: bool = True,
                         outfile="dynamic_results.json"):
    """
    适用：页面通过 JS 加载内容（XHR / fetch / CSR）
    需先安装：pip install playwright && playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("需要安装 playwright：pip install playwright && playwright install chromium")

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=pick_ua(), viewport={"width": 1280, "height": 900})
        page = await context.new_page()
        for url in urls:
            logger.info("打开: %s", url)
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if resp and resp.status != 200:
                    logger.warning("状态码 %s => %s", url, resp.status)
                if wait_selector:
                    try:
                        await page.wait_for_selector(wait_selector, timeout=15000)
                    except Exception:
                        logger.warning("等待选择器超时: %s", wait_selector)
                if scroll:
                    # 模拟增量滚动加载
                    for _ in range(6):
                        await page.mouse.wheel(0, 1200)
                        await asyncio.sleep(0.8)
                # 提取（示例：抓取所有含 data-item 的块）
                items = await page.eval_on_selector_all(
                    "div,li,article",
                    """
                    (nodes) => nodes
                      .filter(n => n.innerText && n.innerText.length > 30)
                      .slice(0, 20)
                      .map(n => ({text: n.innerText.slice(0,300)}))
                    """
                )
                results.append({
                    "url": url,
                    "items": items,
                    "title": await page.title()
                })
            except Exception as e:
                logger.error("动态抓取失败 %s err=%s", url, e)
        await browser.close()
    Path(outfile).write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    logger.info("动态抓取完成, 写出 %s", outfile)

# ===========================================================
# d) Sitemap 增量抓取
# ===========================================================
def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    简单解析 <loc>URL</loc>
    """
    logger.info("下载 sitemap: %s", sitemap_url)
    r = requests.get(sitemap_url, headers={"User-Agent": pick_ua()}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"下载失败: {r.status_code}")
    locs = re.findall(r"<loc>(.*?)</loc>", r.text)
    return list(dict.fromkeys(locs))

def sitemap_scrape(sitemap_url: str, state_file="sitemap_state.json", limit: int = 30):
    urls = parse_sitemap(sitemap_url)
    logger.info("Sitemap URL 数: %d", len(urls))
    prev = {}
    if Path(state_file).exists():
        with contextlib.suppress(Exception):
            prev = json.loads(Path(state_file).read_text("utf-8"))

    visited = set(prev.get("visited", []))
    new_urls = [u for u in urls if u not in visited][:limit]
    logger.info("增量新 URL: %d (limit=%d)", len(new_urls), limit)
    if not new_urls:
        logger.info("无新增，结束")
        return

    cache = SimpleCache(ttl=86400)
    store = ArticleStore()

    for u in new_urls:
        c = cache.get(u)
        if c:
            html = c["html"]
        else:
            time.sleep(0.5)
            resp = requests.get(u, headers={"User-Agent": pick_ua()}, timeout=15)
            if resp.status_code != 200:
                logger.warning("跳过 %s 状态=%s", u, resp.status_code)
                continue
            html = resp.text
            cache.set(u, {"html": html, "fetched_at": time.time()})

        soup = BeautifulSoup(html, "lxml")
        title_el = soup.select_one("title")
        title = title_el.get_text(strip=True) if title_el else ""
        body_text = " ".join(p.get_text(" ", strip=True) for p in soup.select("p")[:40])
        store.upsert({
            "url": u,
            "title": title,
            "author": "",
            "published": "",
            "summary": body_text[:160],
            "content": body_text,
            "raw_html": html[:200000]
        })
        visited.add(u)
        logger.info("保存: %s", title[:40])

    Path(state_file).write_text(json.dumps({"visited": list(visited)}, ensure_ascii=False, indent=2), "utf-8")
    logger.info("状态写入: %s", state_file)

# ===========================================================
# 命令入口
# ===========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="多场景合规采集脚本集合 (选择一个模式执行)")
    sub = parser.add_subparsers(dest="cmd")

    p_static = sub.add_parser("static", help="静态页面抓取示例")
    p_static.add_argument("--urls", nargs="+", required=True, help="种子 URL 列表")
    p_static.add_argument("--rate", type=float, default=1.5, help="每秒请求数 (默认1.5)")
    p_static.add_argument("--ttl", type=int, default=3600, help="缓存秒数 (默认1h)")

    p_api = sub.add_parser("api", help="分页 API 抓取")
    p_api.add_argument("--base-url", required=True, help="包含 {page} 占位符的 URL")
    p_api.add_argument("--start", type=int, default=1)
    p_api.add_argument("--pages", type=int, default=5)
    p_api.add_argument("--concurrency", type=int, default=5)
    p_api.add_argument("--delay", type=float, default=0.2)
    p_api.add_argument("--out", default="api_data.json")

    p_dyn = sub.add_parser("dynamic", help="动态渲染抓取 (Playwright)")
    p_dyn.add_argument("--urls", nargs="+", required=True)
    p_dyn.add_argument("--wait", help="等待出现的 CSS 选择器")
    p_dyn.add_argument("--scroll", action="store_true", help="模拟滚动加载")
    p_dyn.add_argument("--no-headless", action="store_true")
    p_dyn.add_argument("--out", default="dynamic_results.json")

    p_site = sub.add_parser("sitemap", help="Sitemap 增量抓取")
    p_site.add_argument("--sitemap", required=True)
    p_site.add_argument("--state", default="sitemap_state.json")
    p_site.add_argument("--limit", type=int, default=30, help="每次新增处理上限")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == "static":
        static_scrape(args.urls, cache_ttl=args.ttl, rate=args.rate)
    elif args.cmd == "api":
        asyncio.run(api_crawler(
            base_url=args.base_url,
            start=args.start,
            limit_pages=args.pages,
            concurrency=args.concurrency,
            delay=args.delay,
            outfile=args.out
        ))
    elif args.cmd == "dynamic":
        asyncio.run(dynamic_scrape(
            urls=args.urls,
            wait_selector=args.wait,
            scroll=args.scroll,
            headless=not args.no_headless,
            outfile=args.out
        ))
    elif args.cmd == "sitemap":
        sitemap_scrape(
            sitemap_url=args.sitemap,
            state_file=args.state,
            limit=args.limit
        )

if __name__ == "__main__":
    main()
