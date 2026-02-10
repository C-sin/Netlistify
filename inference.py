import shutil

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image
from shapely.geometry import box, point

from netlistify_export import export_json

import json
from pathlib import Path
from util import cut_box

from slice import *
from testing import *
from util import *
from utility import *

config_flag = False
device_name = "cpu"
device = torch.device(device_name)

# 加载权重时的 map_location 也会跟着用 CPU
model_orientation = torch.load(
    "bubble_orientation/res50_1.pt", weights_only=False, map_location=device_name
)
model_bubble = torch.load(
    "bubble_orientation/cc_res50_1.pt", weights_only=False, map_location=device_name
)

# Load the model
if config_flag:
    raise NotImplementedError("Please train the correponding model from the provided dataset first.")
else:
    model_orientation = torch.load(
        "bubble_orientation/res50_1.pt", weights_only=False, map_location=device_name
    )
model_bubble = torch.load(
    "bubble_orientation/cc_res50_1.pt", weights_only=False, map_location=device_name
)


def get_box(path, filename):
    with open(path + "labels/" + filename + ".txt", "r") as txt:
        lines = txt.readlines()
        all_cls = []
        all_xyxyn = []
        for line in lines:
            line = line.split()
            cls = int(line[0])
            all_cls.append(cls)
            x_yolo = float(line[1])
            y_yolo = float(line[2])
            yolo_width = float(line[3])
            yolo_height = float(line[4])

            # Convert Yolo Format to Pascal VOC format
            box_width = yolo_width
            box_height = yolo_height
            x_min = x_yolo - (box_width / 2)
            y_min = y_yolo - (box_height / 2)
            x_max = x_yolo + (box_width / 2)
            y_max = y_yolo + (box_height / 2)
            all_xyxyn.append([x_min, y_min, x_max, y_max])

        return all_cls, all_xyxyn


class component:
    def __init__(self, type, pos):
        self.type = type
        self.pos = pos
        self.orientation = None
        self.bubble = None
        self.pin = []


def get_predictions(model, input_tensors):
    input_batch = torch.stack(input_tensors).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(input_batch)  # 输出形状 (batch_size, num_classes)

    y_prob = F.softmax(y_pred, dim=-1)
    top_preds = y_prob.argmax(1)
    return top_preds


def change_axis(group_connection, shape):
    for i in range(len(group_connection)):
        for j in range(len(group_connection[i])):
            group_connection[i][j] = (
                group_connection[i][j][0] * shape[1],
                shape[0] - group_connection[i][j][1] * shape[0],
            )
    return group_connection


def gnd_transfer(comps):
    gnd_net = []
    for i in comps:
        if comps[i].type == "gnd":
            gnd_net.append(comps[i].pin[0])

    for i in comps:
        for index, j in enumerate(comps[i].pin):
            if j in gnd_net:
                comps[i].pin[index] = "gnd"
    return comps


comp_table = {
    "pmos": "pmos4",
    "nmos": "nmos4",
    "resistor": "r",
    "capacity": "c",
    "inductor": "l",
    "voltage": "v",
    "current": "i",
    "diode": "diode",
    "not": "inverter",
    "op": "op",
    "tgate": "tgate",
    "npn": "npn",
    "pnp": "pnp",
    "func": "dflipflop",
}


def write_sp(filename, comps, path):
    file_path = Path(path) / f"{filename}.sp"
    print("write to:", file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write(f".subckt {filename}\n")

        # 在子电路头部写一段坐标索引的注释块
        f.write("* === component bounding boxes (pixel coords) ===\n")
        for name, c in comps.items():
            x1, y1, x2, y2 = c.pos
            f.write(f"* {name} type={c.type} bbox=({x1},{y1},{x2},{y2}) orientation={c.orientation}\n")
            # 可选：写每个引脚的小框坐标
            rects = cut_box(False, c.type, c.pos, c.orientation)
            for p_idx, r in enumerate(rects):
                rx1, ry1, rx2, ry2 = r
                f.write(f"*   pin{p_idx}_rect=({rx1},{ry1},{rx2},{ry2})\n")

        # 正式的 netlist 元件行
        for name, c in comps.items():
            if c.type == "gnd":
                continue
            elif c.type in ["or", "xor", "and"]:
                f.write(name + " ")
                for j in c.pin:
                    f.write(j + " ")
                if c.bubble is not None and c.bubble == 0:
                    f.write("n" + c.type + str(len(c.pin) - 1))
                else:
                    f.write(c.type + str(len(c.pin) - 1))
                f.write("\n")
            else:
                f.write(name + " ")
                for j in c.pin:
                    f.write(j + " ")
                from inference import comp_table 
                f.write(comp_table[c.type])
                f.write("\n")

        f.write(".ends\n")

def write_component_meta(filename, comps, path):
    meta = []
    for name, c in comps.items():
        entry = {
            "name": name,
            "type": c.type,
            "bbox": {"x1": c.pos[0], "y1": c.pos[1], "x2": c.pos[2], "y2": c.pos[3]},
            "orientation": c.orientation,
            "bubble": c.bubble,
            "pins": []
        }
        rects = cut_box(False, c.type, c.pos, c.orientation)
        for idx, r in enumerate(rects):
            entry["pins"].append({
                "index": idx,
                "rect": {"x1": r[0], "y1": r[1], "x2": r[2], "y2": r[3]}
            })
        meta.append(entry)

    out_path = Path(path) / f"{filename}_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("write meta to:", out_path)

transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


if config_flag:
    type = [
        "gnd",
        "pmos",
        "nmos",
        "pnp",
        "npn",
        "resistor",
        "capacity",
        "voltage",
        "current",
        "diode",
        "inductor",
        "and",
        "or",
        "xor",
        "not",
        "func",
        "op",
        "tgate",
    ]
else:
    type = [
        "gnd",
        "pmos",
        "nmos",
        "pnp",
        "npn",
        "resistor",
        "capacity",
        "voltage",
        "current",
        "text",
        "node",
        "crossing",
    ]

type_name = [
    "gnd",
    "m",
    "m",
    "q",
    "q",
    "r",
    "c",
    "v",
    "i",
    "d",
    "l",
    "xi",
    "xi",
    "xi",
    "xi",
    "xi",
    "xi",
    "xi",
]

type_need_orientation = [
    "pmos",
    "nmos",
    "pnp",
    "npn",
    "voltage",
    "current",
    "diode",
    "and",
    "or",
    "xor",
    "not",
    "op",
    "tgate",
]

type_need_bubble = ["and", "or"]

orientation_cls = ["MX", "MXR90", "MY", "MYR90", "R0", "R180", "R270", "R90"]


def inference(img_path, output_folder):
    label_path = f"{Path(img_path).parents[1]}/"
    filename = Path(img_path).stem
    comps = {}
    img = cv2.imread(img_path)
    cv2.imwrite(output_folder + f"{filename}_input.jpg", img)
    img_h, img_w, _ = img.shape
    imgshow = img.copy()
    no = 0
    input_tensors = []
    input_tensors_bubble = []
    orientation_no = []
    bubble_no = []
    all_xyxy = []
    all_xyxyn = []
    all_cls, all_xyxyn = get_box(label_path, filename)

    if img_h > 870 or img_w > 870:
        img = cv2.resize(img, (int(img_w / 3), int(img_h / 3)))
        img_h, img_w, _ = img.shape
        imgshow = img.copy()
        all_xyxy.extend(
            [
                (
                    int(xyxyn[0] * img_w),
                    int(xyxyn[1] * img_h),
                    int(xyxyn[2] * img_w),
                    int(xyxyn[3] * img_h),
                )
                for xyxyn in all_xyxyn
            ]
        )
    else:
        all_xyxy.extend(
            [
                (
                    int(xyxyn[0] * img_w),
                    int(xyxyn[1] * img_h),
                    int(xyxyn[2] * img_w),
                    int(xyxyn[3] * img_h),
                )
                for xyxyn in all_xyxyn
            ]
        )
    for instance in zip(all_cls, all_xyxy):
        cls, xyxy = instance
        if type[cls] in ["text", "node", "crossing"] and not config_flag:
            continue
        else:
            name = type_name[cls] + str(no)
            comps[name] = component(type[cls], xyxy)
            cv2.rectangle(imgshow, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
        no += 1

    for i in comps:
        if comps[i].type in type_need_orientation:
            x1, y1, x2, y2 = comps[i].pos
            img_crop = img[y1:y2, x1:x2]
            crop_img_pil = Image.fromarray(
                cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            )  # Convert to PIL image
            input_tensor = transform(crop_img_pil)  # Apply preprocessing
            input_tensors.append(input_tensor)
            orientation_no.append(i)
            if comps[i].type in type_need_bubble:
                input_tensors_bubble.append(input_tensor)
                bubble_no.append(i)

    if len(input_tensors) > 0:
        top_preds = get_predictions(model_orientation, input_tensors)
        top_preds = [orientation_cls[i] for i in top_preds]
    else:
        top_preds = []
    # print(top_preds)
    if len(input_tensors_bubble) > 0:
        bubble_preds = get_predictions(model_bubble, input_tensors_bubble)
        bubble_preds = [int(i) for i in bubble_preds]
    else:
        bubble_preds = []

    for i in range(len(orientation_no)):
        comps[orientation_no[i]].orientation = top_preds[i]
    for i in range(len(bubble_no)):
        comps[bubble_no[i]].bubble = bubble_preds[i]

    if config_flag:
        config_ = config.DatasetConfig.CC
    else:
        config_ = config.DatasetConfig.REAL

    img, processed_img = load_test_data(img, filename + ".txt", label_path, config_)
    group_connection, result_img = analyze_connection(
        config_,
        processed_img.copy(),
        img.copy(),
        # min_line_length=1e-4,
        interval=5,
        local_threshold=0.13,
        global_threshold=0.13,
        # strict_match=True,
        # strict_match_threshold=0.15,
        debug=True,
        debug_cell=[-1, -1],
        debug_img_width=600,
        output_folder=str(output_folder),
    )

    group_connection = change_axis(group_connection, img.shape[:2])

    connection_dict = {}
    net_cnt = 0

    for i in group_connection:
        net_name = "net" + str(net_cnt)
        connection_dict[net_name] = i
        net_cnt += 1
    for i in comps:
        rect = box(comps[i].pos[0], comps[i].pos[1], comps[i].pos[2], comps[i].pos[3])
        rects = cut_box(config_flag, comps[i].type, comps[i].pos, comps[i].orientation)
        for index, j in enumerate(rects):
            rect = box(j[0], j[1], j[2], j[3])
            pins_found = []
            if len(rects) == 1:
                rect = box(rects[0][0], rects[0][1], rects[0][2], rects[0][3])
                if comps[i].type == "gnd":
                    min_distance = 10
                    candidates = None
                    for k, points in connection_dict.items():
                        for l in points:
                            dist = rect.distance(point.Point(l))
                            if dist < min_distance:
                                candidates = k
                                min_distance = dist
                    comps[i].pin.append(candidates)
                else:
                    for k, points in connection_dict.items():
                        for l in points:
                            pt = point.Point(l)
                            if rect.contains(pt):
                                comps[i].pin.append(k)
            else:
                if comps[i].type in ["or", "xor", "and"] and index == 0:
                    # print(rect)
                    for k, points in connection_dict.items():
                        match_count = 0  # 计数器初始化为0
                        for l in points:
                            if rect.contains(point.Point(l)):
                                pins_found.append(k)
                                match_count += 1  # 每次匹配增加计数
                                if match_count >= 3:  # 达到3次匹配则停止添加
                                    break

                    comps[i].pin.extend(pins_found)
                    comps[i].pin.extend(
                        ["net" + str(i + net_cnt) for i in range(2 - len(pins_found))]
                    )
                else:
                    min_distance = float("inf")
                    candidates = None
                    for k, points in connection_dict.items():
                        for l in points:
                            dist = rect.distance(point.Point(l))
                            if dist < min_distance:
                                candidates = k
                                min_distance = dist
                    for l in comps:
                        if comps[l].type == "gnd":
                            gnd_box = box(
                                comps[l].pos[0], comps[l].pos[1], comps[l].pos[2], comps[l].pos[3]
                            )
                            if gnd_box.distance(rect) < min_distance:
                                candidates = "gnd"
                                min_distance = gnd_box.distance(rect)
                    if candidates:
                        comps[i].pin.append(candidates)
                    else:
                        comps[i].pin.append("net" + str(net_cnt))
                        if comps[i].type in ["nmos", "pmos"] and index == 2:
                            pass
                        else:
                            net_cnt += 1
    comps = gnd_transfer(comps)
    write_sp(filename + "_output", comps, output_folder)
    write_component_meta(filename + "_output", comps, output_folder)
    export_json(filename + "_output", comps, connection_dict, output_folder)
    return comps


is_code_ocean = False
img_dir = "test_images/images/" if not is_code_ocean else "/data/test_images/images/"
output_folder = Path("/results/" if is_code_ocean else "./results/")

# Clear all content in output_folder but don't remove output_folder itself
if output_folder.exists():
    for item in output_folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
output_folder.mkdir(exist_ok=True)

for img_path in list(Path(img_dir).iterdir()):
    img_path = str(img_path)

    output_img_dir = output_folder / Path(Path(img_path).stem)
    if output_img_dir.exists():
        shutil.rmtree(output_img_dir)
    output_img_dir.mkdir()
    inference(img_path, output_folder=f"{output_img_dir}/")

print("Inference completed successfully.")
print("========================================")
print(
    "The output netlist of the image are created in the results tab, which is xxx_input.jpg and xxx_output.sp. We also saved the middle results such as model_predictions, connection of components and the attention map in the results folder."
)
print("Thank you for using our inference script!")
