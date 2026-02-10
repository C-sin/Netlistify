import argparse
from pathlib import Path
from inference import inference  # 调用现有的单图推理函数

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入电路图路径（绝对或相对）")
    parser.add_argument("--out", default="./results", help="输出目录（默认 ./results）")
    args = parser.parse_args()

    img_path = Path(args.image).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    out_dir = out_root / stem
    if out_dir.exists():
        # 清理上一次的结果
        for p in out_dir.iterdir():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 调用 Netlistify 的核心推理函数
    inference(str(img_path), output_folder=str(out_dir) + "/")

    print("OK")
    print("sp:", out_dir / f"{stem}_output.sp")
    # 如果你已集成 JSON 导出（export_json）
    print("json:", out_dir / f"{stem}_output_graph.json")

if __name__ == "__main__":
    main()