import sys
from rknn.api import RKNN

DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = '../model/yolo11.rknn'
DEFAULT_QUANT = True

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path platform [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("   or: python3 {} onnx_model_path platform output_rknn_path".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT

    dtype_choices = ['i8', 'u8', 'fp']
    output_path = DEFAULT_RKNN_PATH

    if len(sys.argv) == 4:
        arg3 = sys.argv[3]
        if arg3 in dtype_choices:
            do_quant = arg3 in ['i8', 'u8']
        else:
            output_path = arg3
    elif len(sys.argv) >= 5:
        arg3 = sys.argv[3]
        arg4 = sys.argv[4]
        if arg3 in dtype_choices:
            do_quant = arg3 in ['i8', 'u8']
            output_path = arg4
        elif arg4 in dtype_choices:
            output_path = arg3
            do_quant = arg4 in ['i8', 'u8']
        else:
            print("ERROR: Invalid dtype: {}".format(arg3))
            exit(1)

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
