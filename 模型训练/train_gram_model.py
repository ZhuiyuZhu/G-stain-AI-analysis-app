# -*- coding: utf-8 -*-
"""
革兰染色涂片检测模型训练脚本（YOLO）
用法：
  1. pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
  2. 按 data.yaml 中的目录结构放好 图像+标注
  3. python train_gram_model.py
  4. 训练完成后把 runs/detect/gram_train/weights/best.pt
     重命名为 gram_model.pt，放到 gram_gui.py 同目录即可启用AI模式
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml", help="数据集配置文件")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="预训练权重：yolo11n.pt(快/轻量) yolo11s.pt(更准) 或 yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="输入尺寸，细菌目标小，建议>=1280")
    ap.add_argument("--batch", type=int, default=8,
                    help="显存不足改小(4/2)；纯CPU训练建议 imgsz 640 batch 4")
    ap.add_argument("--device", default="0",
                    help="GPU编号；无独显填 cpu（很慢，仅适合试跑）")
    ap.add_argument("--resume", default="", help="中断续训：填 last.pt 路径")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="gram_train",
        # ---- 针对显微图像的数据扩增 ----
        hsv_h=0.03,       # 色相扰动：模拟染色批次差异（不宜过大，否则阴阳性混淆）
        hsv_s=0.4,        # 饱和度扰动：模拟染色深浅
        hsv_v=0.4,        # 明度扰动：模拟曝光差异
        degrees=180,      # 旋转：涂片无方向性
        flipud=0.5,
        fliplr=0.5,
        scale=0.3,
        mosaic=1.0,
        copy_paste=0.1,   # 复制粘贴扩增：缓解类别不均衡
        # ---- 训练策略 ----
        patience=30,      # 早停
        cos_lr=True,
        workers=4,
    )

    # 验证集评估
    metrics = model.val()
    print("\n===== 验证集指标 =====")
    print(f"mAP50:    {metrics.box.map50:.3f}  （>0.6 可试用，>0.75 较好）")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    # 导出ONNX（可选：不想装ultralytics的运行环境用）
    model.export(format="onnx", imgsz=args.imgsz)
    print("\n训练完成。把 runs/detect/gram_train/weights/best.pt")
    print("重命名为 gram_model.pt 放到软件目录即可。")


if __name__ == "__main__":
    main()
