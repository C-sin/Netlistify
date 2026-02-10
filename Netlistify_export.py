import json
from pathlib import Path
from typing import Dict, Any, List

def export_json(filename: str,
                comps: Dict[str, Any],
                connection_dict: Dict[str, List[List[float]]],
                out_dir: str):
    """
    导出结构化 JSON，包含：
    - components：每个元件的类型、bbox、orientation、bubble、pins（含 net 和引脚小框 rect）
    - nets：每个网络的几何点（用于可视化）
    """
    data = {
        "subckt": filename,
        "components": [],
        "nets": [],
    }

    # 组件与引脚
    for name, c in comps.items():
        comp_entry = {
            "name": name,
            "type": c.type,
            "bbox": {"x1": int(c.pos[0]), "y1": int(c.pos[1]), "x2": int(c.pos[2]), "y2": int(c.pos[3])},
            "orientation": c.orientation,
            "bubble": c.bubble,
            "pins": []
        }
        # 引脚的网络映射（按推理阶段生成的 c.pin）
        for idx, net in enumerate(c.pin):
            comp_entry["pins"].append({
                "index": idx,
                "net": net
            })
        data["components"].append(comp_entry)

    # 网络的几何点集（来自 connection_dict）
    for net_name, points in connection_dict.items():
        data["nets"].append({
            "net": net_name,
            "points": points  # 这里是像素坐标点（x,y），用于绘制连接线
        })

    out_path = Path(out_dir) / f"{filename}_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("write json to:", out_path)