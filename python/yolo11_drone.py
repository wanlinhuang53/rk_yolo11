import os
import cv2
import sys
import argparse
import numpy as np
from rknn.api import RKNN

script_dir = os.path.dirname(os.path.abspath(__file__))
_sep = os.path.sep
_added_py_utils = False

_env_path = os.environ.get('PY_UTILS_PATH') or os.environ.get('RKNN_MODEL_ZOO_PATH')
_candidates = []
if _env_path:
    _candidates.append(_env_path)

_candidates.extend([
    os.path.abspath(os.path.join(script_dir, '..', '..')),
    os.path.abspath(os.path.join(script_dir, '..', '..', 'py_utils')),
    os.path.abspath(os.path.join(script_dir, '..', '..', '..')),
    os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'py_utils')),
    os.path.abspath(os.path.join(script_dir, '..', '..', 'rknn_model_zoo')),
    os.path.abspath(os.path.join(script_dir, '..', '..', 'rknn_model_zoo', 'py_utils')),
])

for _p in _candidates:
    if not _p:
        continue
    if os.path.isdir(os.path.join(_p, 'py_utils')):
        sys.path.append(_p)
        _added_py_utils = True
        break
    if os.path.isdir(_p) and os.path.basename(_p) == 'py_utils':
        sys.path.append(os.path.dirname(_p))
        _added_py_utils = True
        break

if not _added_py_utils:
    realpath = os.path.abspath(__file__).split(_sep)
    try:
        sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('rknn_model_zoo') + 1]))
    except ValueError:
        raise RuntimeError(
            "Cannot locate 'py_utils'. Set PY_UTILS_PATH (parent dir containing py_utils) or RKNN_MODEL_ZOO_PATH, "
            "or place 'py_utils' under the project root."
        )

from py_utils.coco_utils import COCO_test_helper

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CLASSES = ("drone",)


def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)


def dfl(position):
    x = position.astype(np.float32)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y - np.max(y, axis=2, keepdims=True)
    y = np.exp(y)
    y = y / np.sum(y, axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(axis=2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cls_name = CLASSES[int(cl)] if int(cl) < len(CLASSES) else str(int(cl))
        print("%s @ (%d %d %d %d) %.3f" % (cls_name, top, left, right, bottom, float(score)))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(cls_name, float(score)), (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--rknn', type=str, default=None)
    parser.add_argument('--platform', type=str, default='rk3588')
    parser.add_argument('--optimization_level', type=int, default=3)
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
    parser.add_argument('--do_quant', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--accuracy_analysis', action='store_true', default=False)
    parser.add_argument('--analysis_inputs', type=str, default=None)
    parser.add_argument('--snapshot_dir', type=str, default='snapshot')
    parser.add_argument('--analysis_limit', type=int, default=0)
    parser.add_argument('--img_save', action='store_true', default=False)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError('ONNX model not found: {}'.format(args.onnx))
    if not os.path.exists(args.img):
        raise FileNotFoundError('Image not found: {}'.format(args.img))

    if args.rknn is None:
        args.rknn = os.path.splitext(args.onnx)[0] + '.rknn'

    if args.do_quant and not args.dataset:
        raise ValueError('Quantization enabled but --dataset is not set')

    if args.accuracy_analysis and args.analysis_inputs is None:
        if args.dataset:
            args.analysis_inputs = args.dataset
        else:
            args.analysis_inputs = args.img

    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        optimization_level=args.optimization_level,
    )
    print('done')

    print('--> Loading onnx model')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError('load_onnx failed: {}'.format(ret))
    print('done')

    print('--> Building model')
    build_kwargs = {'do_quantization': args.do_quant}
    if args.do_quant:
        build_kwargs['dataset'] = args.dataset
    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError('build failed: {}'.format(ret))
    print('done')

    if args.accuracy_analysis:
        inputs = []
        p = args.analysis_inputs
        if os.path.isdir(p):
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            for dirpath, _, filenames in os.walk(p):
                for name in filenames:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in exts:
                        inputs.append(os.path.join(dirpath, name))
        elif p.lower().endswith('.txt') and os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        inputs.append(line)
        else:
            inputs.append(p)

        if args.analysis_limit and args.analysis_limit > 0:
            inputs = inputs[:args.analysis_limit]

        if not inputs:
            raise RuntimeError('accuracy_analysis has no inputs: {}'.format(args.analysis_inputs))

        print('--> Accuracy analysis (simulator)')
        try:
            ret = rknn.accuracy_analysis(inputs=inputs, output_dir=args.snapshot_dir)
        except TypeError:
            ret = rknn.accuracy_analysis(inputs, args.snapshot_dir)
        if ret != 0:
            raise RuntimeError('accuracy_analysis failed: {}'.format(ret))
        print('done. Snapshot saved to {}'.format(args.snapshot_dir))
        rknn.release()
        return

    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn)
    if ret != 0:
        raise RuntimeError('export_rknn failed: {}'.format(ret))
    print('done')

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
        raise RuntimeError('init_runtime failed: {}'.format(ret))
    print('done')

    if args.eval_perf and hasattr(rknn, 'eval_perf') and args.target is not None:
        try:
            rknn.eval_perf(args.is_print, args.fix_freq)
        except Exception:
            pass

    if args.eval_memory and hasattr(rknn, 'eval_memory') and args.target is not None:
        try:
            rknn.eval_memory(args.is_print)
        except Exception:
            pass

    img_src = cv2.imread(args.img)
    if img_src is None:
        raise RuntimeError('cv2.imread failed: {}'.format(args.img))

    co_helper = COCO_test_helper(enable_letter_box=True)
    img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img, axis=0)

    def _infer_once():
        try:
            return rknn.inference(inputs=[input_data], data_format='nhwc')
        except TypeError:
            return rknn.inference(inputs=[input_data])

    for _ in range(max(0, args.warmup)):
        _infer_once()

    outputs = None
    for _ in range(max(1, args.repeat)):
        outputs = _infer_once()

    boxes, classes, scores = post_process(outputs)

    if boxes is None:
        print('No drone detected')
        rknn.release()
        return

    img_p = img_src.copy()
    draw(img_p, co_helper.get_real_box(boxes), scores, classes)

    if args.img_save:
        out_path = args.out
        if out_path is None:
            base = os.path.splitext(os.path.basename(args.img))[0]
            out_path = os.path.join(os.path.dirname(__file__), 'result_sim_drone', '{}.jpg'.format(base))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img_p)
        print('Detection result save to {}'.format(out_path))

    rknn.release()


if __name__ == '__main__':
    main()
