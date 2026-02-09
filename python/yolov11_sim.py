import os
import argparse
import time
import cv2
import numpy as np
from rknn.api import RKNN

# Reuse YOLO11 postprocess/draw/letterbox helper from existing demo
from yolo11 import COCO_test_helper, IMG_SIZE, post_process, draw

# Paths (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL = os.path.join(SCRIPT_DIR, '..', 'model', 'yolo11n.onnx')
IMG_PATH = os.path.join(SCRIPT_DIR, '..', 'model', 'bus.jpg')
RKNN_MODEL = os.path.join(SCRIPT_DIR, '..', 'model', 'yolo11_sim.rknn')
OUT_DIR = os.path.join(SCRIPT_DIR, 'result_sim')

# Simulator inference: must use load_onnx -> build -> init_runtime(target=None)
TARGET_PLATFORM = 'rk3588'
DO_QUANT = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default=ONNX_MODEL)
    parser.add_argument('--img', default=IMG_PATH)
    parser.add_argument('--rknn', default=RKNN_MODEL)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--perf_debug', action='store_true', default=False)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--eval_perf', action='store_true', default=False)
    parser.add_argument('--eval_memory', action='store_true', default=False)
    parser.add_argument('--is_print', action='store_true', default=True)
    parser.add_argument('--no_print', dest='is_print', action='store_false')
    parser.add_argument('--fix_freq', action='store_true', default=True)
    parser.add_argument('--no_fix_freq', dest='fix_freq', action='store_false')
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f'ONNX model not found: {args.onnx}')
    if not os.path.exists(args.img):
        raise FileNotFoundError(f'Image not found: {args.img}')

    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=TARGET_PLATFORM)
    print('done')

    print('--> Loading onnx model')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError(f'load_onnx failed: {ret}')
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=DO_QUANT)
    if ret != 0:
        raise RuntimeError(f'build failed: {ret}')
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn)
    if ret != 0:
        raise RuntimeError(f'export_rknn failed: {ret}')
    print(f'done. RKNN model saved to {args.rknn}')

    init_kwargs = {
        'target': args.target,
        'device_id': args.device_id,
        'perf_debug': args.perf_debug,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    print('--> Init runtime environment')
    try:
        ret = rknn.init_runtime(**init_kwargs)
    except TypeError:
        init_kwargs.pop('perf_debug', None)
        ret = rknn.init_runtime(**init_kwargs)
    if ret != 0:
        raise RuntimeError(f'init_runtime failed: {ret}')
    print('done')

    img_src = cv2.imread(args.img)
    if img_src is None:
        raise RuntimeError(f'cv2.imread failed: {args.img}')

    co_helper = COCO_test_helper(enable_letter_box=True)

    # Keep consistent with yolo11.py
    img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # For RKNN NHWC uint8 input
    input_data = np.expand_dims(img, axis=0)

    def _infer_once():
        try:
            return rknn.inference(inputs=[input_data], data_format='nhwc')
        except TypeError:
            return rknn.inference(inputs=[input_data])

    if args.eval_perf:
        if not hasattr(rknn, 'eval_perf'):
            print('Warning: rknn.eval_perf is not available in this rknn-toolkit2 version.')
        elif args.target is None:
            print('Warning: eval_perf requires running on a real device. Please set --target and connect the board.')
        else:
            print('--> Eval perf (rknn.eval_perf)')
            try:
                ret = rknn.eval_perf(args.is_print, args.fix_freq)
                if ret is not None:
                    print(ret)
            except Exception as e:
                print(f'Warning: eval_perf failed: {e}')

    if args.eval_memory:
        if not hasattr(rknn, 'eval_memory'):
            print('Warning: rknn.eval_memory is not available in this rknn-toolkit2 version.')
        elif args.target is None:
            print('Warning: eval_memory requires running on a real device. Please set --target and connect the board.')
        else:
            print('--> Eval memory (rknn.eval_memory)')
            try:
                ret = rknn.eval_memory(args.is_print)
                if ret is not None:
                    print(ret)
            except Exception as e:
                print(f'Warning: eval_memory failed: {e}')

    # 推理计时：首次 inference 可能包含 GraphPreparing/SessionPreparing 等一次性开销
    for _ in range(max(0, args.warmup)):
        _infer_once()

    infer_times = []
    outputs = None
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        outputs = _infer_once()
        t1 = time.perf_counter()
        infer_times.append((t1 - t0) * 1000)  # ms

    infer_avg = float(np.mean(infer_times))
    infer_min = float(np.min(infer_times))
    infer_max = float(np.max(infer_times))

    # 后处理计时
    t2 = time.perf_counter()
    boxes, classes, scores = post_process(outputs)
    t3 = time.perf_counter()
    post_time = (t3 - t2) * 1000  # ms

    print(f'Inference time (ms): avg={infer_avg:.2f}, min={infer_min:.2f}, max={infer_max:.2f} (repeat={max(1, args.repeat)}, warmup={max(0, args.warmup)})')
    print(f'Post-process time: {post_time:.2f} ms')
    print(f'Total time: {infer_avg + post_time:.2f} ms')

    img_p = img_src.copy()
    if boxes is not None:
        draw(img_p, co_helper.get_real_box(boxes), scores, classes)

    os.makedirs(OUT_DIR, exist_ok=True)
    # 根据模型名称生成输出图片名
    model_name = os.path.splitext(os.path.basename(args.onnx))[0]
    img_name = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(OUT_DIR, f'{img_name}_{model_name}.jpg')
    cv2.imwrite(out_path, img_p)
    print(f'Detection result save to {out_path}')

    rknn.release()


if __name__ == '__main__':
    main()
