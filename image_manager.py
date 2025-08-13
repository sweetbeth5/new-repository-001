#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线本地图片管理脚本
功能概览：
  1. 扫描目录：收集图片信息（尺寸 / EXIF / Hash / 平均感知哈希）并生成 metadata.json
  2. 去重检测：按文件 SHA256 精确去重；按平均哈希（aHash）阈值检测近似重复
  3. 优化输出：生成多规格 (e.g. 960, 1600) / WebP / (可选 AVIF) 压缩版本，避免重复处理（缓存）
  4. 画廊生成：输出静态 HTML 画廊页面（含懒加载、响应式 srcset）
  5. 缓存机制：.image_cache.json 保存已处理指纹，避免重复计算
  6. CLI：按子命令执行 scan / optimize / gallery / all / clean-cache
依赖：
  pip install pillow tqdm
  （可选 AVIF：pip install pillow-avif-plugin）
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from PIL import Image, ExifTags
except ImportError:
    print("未安装 Pillow，请先执行: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"}
CACHE_FILE = ".image_cache.json"
META_FILE = "metadata.json"
GALLERY_HTML = "gallery.html"

@dataclass
class ImageMeta:
    id: str
    original_path: str
    rel_path: str
    width: int
    height: int
    format: str
    size_bytes: int
    sha256: str
    a_hash: str
    exif: Dict[str, str]
    created_at: str
    processed: Dict[str, Dict[str, str]]  # key -> {path,width,height,size_bytes}
    duplicate_of: Optional[str] = None
    near_duplicates: List[str] = None

def human_size(num: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if num < 1024:
            return f"{num:.1f}{u}"
        num /= 1024
    return f"{num:.1f}TB"

def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def average_hash(img: Image, hash_size: int = 8) -> str:
    """
    产生平均哈希 (aHash) 用于近似重复检测
    返回 64 bit 16进制字符串（8*8 -> 64 bits -> 16 hex）
    """
    try:
        small = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    except Exception:
        small = img.convert("L").resize((hash_size, hash_size))
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return "".join(f"{int(bits[i:i+4], 2):x}" for i in range(0, 64, 4))

def hamming_distance_hex(a: str, b: str) -> int:
    ba = bin(int(a, 16))[2:].zfill(64)
    bb = bin(int(b, 16))[2:].zfill(64)
    return sum(x != y for x, y in zip(ba, bb))

def extract_exif(img: Image) -> Dict[str, str]:
    res = {}
    if not hasattr(img, "_getexif"):
        return res
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return res
        for k, v in exif_raw.items():
            tag = ExifTags.TAGS.get(k, str(k))
            if tag in {"DateTimeOriginal", "Model", "Make", "LensModel", "Software"}:
                vs = str(v)
                if len(vs) > 80:
                    vs = vs[:77] + "..."
                res[tag] = vs
    except Exception:
        pass
    return res

def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache_path: Path, data: Dict):
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)

def scan_images(
    source_dir: Path,
    cache: Dict[str, Dict]
) -> Tuple[List[ImageMeta], Dict[str, List[str]]]:
    images: List[ImageMeta] = []
    sha_index: Dict[str, List[str]] = {}
    file_list = [p for p in source_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXT and p.is_file()]
    iterator = tqdm(file_list, desc="扫描图像") if USE_TQDM else file_list

    for path in iterator:
        rel = path.relative_to(source_dir).as_posix()
        stat = path.stat()
        cache_key = rel
        need_recompute = True
        cache_entry = cache.get(cache_key)
        if cache_entry:
            if cache_entry.get("mtime") == stat.st_mtime_ns and cache_entry.get("size") == stat.st_size:
                try:
                    meta = ImageMeta(**cache_entry["meta"])
                    images.append(meta)
                    sha_index.setdefault(meta.sha256, []).append(meta.id)
                    need_recompute = False
                except Exception:
                    need_recompute = True

        if not need_recompute:
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                fmt = (img.format or path.suffix.replace(".", "")).upper()
                ah = average_hash(img)
                exif = extract_exif(img)
        except Exception as e:
            print(f"[WARN] 无法读取图像: {rel} ({e})")
            continue

        sha = sha256_file(path)
        img_id = sha[:12]

        meta = ImageMeta(
            id=img_id,
            original_path=path.as_posix(),
            rel_path=rel,
            width=width,
            height=height,
            format=fmt,
            size_bytes=stat.st_size,
            sha256=sha,
            a_hash=ah,
            exif=exif,
            created_at=datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            processed={},
            duplicate_of=None,
            near_duplicates=[]
        )
        images.append(meta)
        sha_index.setdefault(sha, []).append(img_id)

        cache[cache_key] = {
            "mtime": stat.st_mtime_ns,
            "size": stat.st_size,
            "meta": asdict(meta)
        }
    return images, sha_index

def detect_duplicates(images: List[ImageMeta], sha_index: Dict[str, List[str]], ahash_threshold: int):
    dup_map = {}
    for sha, ids in sha_index.items():
        if len(ids) > 1:
            primary = ids[0]
            for i in ids[1:]:
                dup_map[i] = primary
    id_map = {m.id: m for m in images}
    for img_id, primary in dup_map.items():
        id_map[img_id].duplicate_of = primary

    valid = [m for m in images if not m.duplicate_of]
    for i, a in enumerate(valid):
        for b in valid[i+1:]:
            dist = hamming_distance_hex(a.a_hash, b.a_hash)
            if dist <= ahash_threshold:
                a.near_duplicates.append(b.id)
                b.near_duplicates.append(a.id)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def optimize_single(
    meta: ImageMeta,
    source_dir: Path,
    out_dir: Path,
    sizes: List[int],
    quality_web: int,
    enable_webp: bool,
    enable_avif: bool
) -> ImageMeta:
    if meta.duplicate_of:
        return meta
    src_path = source_dir / meta.rel_path
    try:
        with Image.open(src_path) as img:
            img.load()
            for target_w in sizes:
                if meta.width <= target_w:
                    key = f"orig_w<= {target_w}"
                    meta.processed.setdefault(key, {
                        "path": meta.rel_path,
                        "width": meta.width,
                        "height": meta.height,
                        "size_bytes": meta.size_bytes
                    })
                    continue
                ratio = target_w / meta.width
                target_h = int(meta.height * ratio)
                resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                base_name = re.sub(r"\.[^.]+$", "", Path(meta.rel_path).name)
                sub_dir = out_dir / Path(meta.rel_path).parent
                ensure_dir(sub_dir)
                std_path = sub_dir / f"{base_name}-{target_w}.jpg"
                if not std_path.exists():
                    try:
                        resized.save(std_path, "JPEG", quality=quality_web, optimize=True, progressive=True)
                    except OSError:
                        resized.convert("RGB").save(std_path, "JPEG", quality=quality_web, optimize=True, progressive=True)
                meta.processed[f"jpg_{target_w}"] = {
                    "path": std_path.relative_to(out_dir).as_posix(),
                    "width": target_w,
                    "height": target_h,
                    "size_bytes": std_path.stat().st_size
                }
                if enable_webp:
                    webp_path = sub_dir / f"{base_name}-{target_w}.webp"
                    if not webp_path.exists():
                        resized.save(webp_path, "WEBP", quality=quality_web, method=6)
                    meta.processed[f"webp_{target_w}"] = {
                        "path": webp_path.relative_to(out_dir).as_posix(),
                        "width": target_w,
                        "height": target_h,
                        "size_bytes": webp_path.stat().st_size
                    }
                if enable_avif:
                    try:
                        avif_path = sub_dir / f"{base_name}-{target_w}.avif"
                        if not avif_path.exists():
                            resized.save(avif_path, "AVIF", quality=quality_web)
                        meta.processed[f"avif_{target_w}"] = {
                            "path": avif_path.relative_to(out_dir).as_posix(),
                            "width": target_w,
                            "height": target_h,
                            "size_bytes": avif_path.stat().st_size
                        }
                    except Exception:
                        pass
    except Exception as e:
        print(f"[WARN] 优化失败: {meta.rel_path} ({e})")
    return meta

def optimize_images(
    images: List[ImageMeta],
    source_dir: Path,
    out_dir: Path,
    sizes: List[int],
    quality_web: int,
    enable_webp: bool,
    enable_avif: bool,
    workers: int
):
    ensure_dir(out_dir)
    tasks = [m for m in images if not m.duplicate_of]
    iterator = tasks
    if USE_TQDM:
        iterator = tqdm(tasks, desc="优化生成")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                optimize_single, m, source_dir, out_dir,
                sizes, quality_web, enable_webp, enable_avif
            ) for m in iterator
        ]
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()

def write_metadata(images: List[ImageMeta], out_path: Path):
    data = [asdict(m) for m in images]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def build_gallery(
    images: List[ImageMeta],
    dist_dir: Path,
    gallery_path: Path,
    sizes: List[int]
):
    ensure_dir(gallery_path.parent)
    rows = []
    for m in images:
        if m.duplicate_of:
            continue
        target_w = min(sizes, key=lambda x: abs(x - 960)) if sizes else m.width
        cand_keys = [k for k in m.processed.keys() if k.endswith(str(target_w))]
        sources_html = ""
        avif = next((m.processed[k] for k in cand_keys if k.startswith("avif_")), None)
        webp = next((m.processed[k] for k in cand_keys if k.startswith("webp_")), None)
        jpeg = next((m.processed[k] for k in cand_keys if k.startswith("jpg_")), None)
        if not jpeg:
            jpeg = {
                "path": m.rel_path,
                "width": m.width,
                "height": m.height,
                "size_bytes": m.size_bytes
            }
        if avif:
            sources_html += f'<source type="image/avif" srcset="optimized/{avif["path"]}" />'
        if webp:
            sources_html += f'<source type="image/webp" srcset="optimized/{webp["path"]}" />'
        rows.append(f"""
        <figure class="item">
          <picture>
            {sources_html}
            <img loading="lazy" src="optimized/{jpeg['path']}" width="{jpeg['width']}" height="{jpeg['height']}" alt="{m.rel_path}" />
          </picture>
          <figcaption>
            <code>{m.rel_path}</code><br/>
            {m.width}×{m.height} · {human_size(m.size_bytes)}
            {" · 重复 => " + m.duplicate_of if m.duplicate_of else ""}
          </figcaption>
        </figure>
        """)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8"/>
  <title>图片画廊 | 离线管理输出</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    body {{
      margin:0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      background:#10151b;
      color:#e6edf3;
    }}
    header {{
      padding:1rem 1.5rem;
      background:#1c2430;
      position:sticky;
      top:0;
      z-index:10;
      display:flex;
      flex-wrap:wrap;
      gap:1rem;
      align-items:center;
    }}
    header h1 {{
      margin:0;
      font-size:1.1rem;
      letter-spacing:.5px;
    }}
    main {{
      padding:1.2rem;
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
      gap:1rem;
    }}
    figure {{
      margin:0;
      background:#1b232e;
      border:1px solid #2b3b4b;
      border-radius:10px;
      overflow:hidden;
      display:flex;
      flex-direction:column;
      box-shadow:0 4px 12px -4px rgba(0,0,0,.4);
      transition:.3s;
    }}
    figure:hover {{
      transform:translateY(-4px);
      border-color:#3b82f6;
    }}
    picture {{
      display:block;
      aspect-ratio:16/10;
      background:#0f141b;
      position:relative;
    }}
    img {{
      width:100%;
      height:100%;
      object-fit:cover;
      display:block;
    }}
    figcaption {{
      font-size:.7rem;
      padding:.55rem .7rem .7rem;
      line-height:1.4;
      word-break:break-all;
      color:#94a3b8;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      background:#243141;
      padding:.15rem .35rem;
      border-radius:4px;
      font-size:.65rem;
    }}
    .stats {{
      font-size:.65rem;
      color:#94a3b8;
      margin-left:auto;
    }}
    footer {{
      text-align:center;
      padding:2rem 1rem 3rem;
      font-size:.65rem;
      color:#64748b;
    }}
    @media (max-width:640px) {{
      main {{
        grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>图片画廊输出</h1>
    <div class="stats">
      总图片: {sum(1 for m in images if not m.duplicate_of)} / 原始(含重复): {len(images)}
    </div>
  </header>
  <main>
    {''.join(rows)}
  </main>
  <footer>
    由离线图片管理脚本生成 · {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")}
  </footer>
</body>
</html>"""
    gallery_path.write_text(html, encoding="utf-8")

def clean_cache(cache_path: Path, meta_path: Path, dist_dir: Path):
    if cache_path.exists():
        cache_path.unlink()
        print(f"已删除缓存: {cache_path}")
    if meta_path.exists():
        meta_path.unlink()
        print(f"已删除元数据: {meta_path}")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print(f"已删除输出目录: {dist_dir}")

def parse_sizes(s: str) -> List[int]:
    return sorted({int(x) for x in re.split(r"[,\s]+", s.strip()) if x})

def main():
    parser = argparse.ArgumentParser(
        description="离线本地图片管理脚本 (扫描 / 去重 / 优化 / 生成画廊)"
    )
    parser.add_argument("command", choices=["scan", "optimize", "gallery", "all", "clean-cache"],
                        help="执行的命令：scan=仅扫描; optimize=仅生成优化图; gallery=仅生成画廊; all=全流程; clean-cache=清理缓存与输出")
    parser.add_argument("-s", "--source", default="images_raw", help="原始图片目录 (默认: images_raw)")
    parser.add_argument("-o", "--output", default="dist", help="输出基目录 (默认: dist)")
    parser.add_argument("--sizes", default="480,960,1600", help="希望生成的宽度列表 (逗号分隔)")
    parser.add_argument("--quality", type=int, default=82, help="JPEG/WebP/AVIF 质量 (默认: 82)")
    parser.add_argument("--no-webp", action="store_true", help="禁用 WebP 输出")
    parser.add_argument("--avif", action="store_true", help="启用 AVIF 输出 (需要 pillow-avif-plugin)")
    parser.add_argument("--ahash-threshold", type=int, default=6, help="平均哈希近似重复距离阈值 (默认:6, 越小越严格)")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数 (默认:4)")
    parser.add_argument("--meta-name", default=META_FILE, help="元数据文件名 (默认: metadata.json)")
    parser.add_argument("--gallery-name", default=GALLERY_HTML, help="画廊文件名 (默认: gallery.html)")
    parser.add_argument("--skip-duplicate-opt", action="store_true", help="跳过重复检测(速度更快)")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 张（调试用）")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dist_base = Path(args.output)
    optimized_dir = dist_base / "optimized"
    meta_path = dist_base / args.meta_name
    gallery_path = dist_base / args.gallery_name
    cache_path = dist_base / CACHE_FILE

    if args.command == "clean-cache":
        clean_cache(cache_path, meta_path, dist_base)
        return

    if not source_dir.exists():
        print(f"[ERROR] 原始目录不存在: {source_dir}", file=sys.stderr)
        sys.exit(2)

    ensure_dir(dist_base)

    sizes = parse_sizes(args.sizes)
    cache = load_cache(cache_path)

    t0 = time.time()
    print("==> 阶段: 扫描图片")
    images, sha_index = scan_images(source_dir, cache)

    if args.limit and args.limit > 0:
        images = images[:args.limit]
        print(f"[调试] 仅保留前 {len(images)} 张进行后续操作")

    if not args.skip_duplicate_opt:
        print("==> 阶段: 去重检测 (SHA + 平均哈希)")
        detect_duplicates(images, sha_index, args.ahash_threshold)
    else:
        print("==> 跳过去重检测")

    if args.command in {"optimize", "all"}:
        print("==> 阶段: 生成优化版本")
        optimize_images(
            images, source_dir, optimized_dir, sizes,
            quality_web=args.quality,
            enable_webp=not args.no-webp,
            enable_avif=args.avif,
            workers=args.workers
        )

    if args.command in {"gallery", "all"}:
        if not optimized_dir.exists():
            print("[WARN] 未发现 optimized 输出目录，画廊将使用原图路径")
        print("==> 阶段: 写出画廊 HTML")
        build_gallery(images, dist_base, gallery_path, sizes)

    print("==> 阶段: 写出元数据")
    write_metadata(images, meta_path)

    for rel, entry in list(cache.items()):
        real_path = source_dir / rel
        if not real_path.exists():
            cache.pop(rel, None)

    for m in images:
        rel = Path(m.rel_path).as_posix()
        p = source_dir / rel
        if not p.exists():
            continue
        st = p.stat()
        cache[rel] = {
            "mtime": st.st_mtime_ns,
            "size": st.st_size,
            "meta": asdict(m)
        }

    save_cache(cache_path, cache)

    elapsed = time.time() - t0
    print("\n================== 汇总 ==================")
    print(f"扫描文件数: {len(images)} (含重复)")
    uniq = sum(1 for m in images if not m.duplicate_of)
    print(f"唯一图片数: {uniq}")
    dups = len(images) - uniq
    print(f"重复图片数: {dups}")
    near_pairs = sum(len(m.near_duplicates) for m in images) // 2
    print(f"近似重复配对数(双向去重): {near_pairs}")
    print(f"元数据文件: {meta_path}")
    if args.command in {"optimize", "all"}:
        print(f"优化目录: {optimized_dir}")
    if args.command in {"gallery", "all"}:
        print(f"画廊页面: {gallery_path}")
    print(f"耗时: {elapsed:.2f}s")
    print("===========================================")

if __name__ == "__main__":
    main()
