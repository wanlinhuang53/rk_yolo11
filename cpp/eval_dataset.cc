#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include "yolo11.h"
#include "file_utils.h"
#include "image_utils.h"

struct BoxF {
    float x1;
    float y1;
    float x2;
    float y2;
};

static std::string trim(const std::string &s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
        b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
        e--;
    return s.substr(b, e - b);
}

static bool is_dir(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

static bool ends_with_lower(const std::string &s, const std::string &suffix_lower)
{
    if (s.size() < suffix_lower.size())
        return false;
    size_t off = s.size() - suffix_lower.size();
    for (size_t i = 0; i < suffix_lower.size(); i++)
    {
        char c = s[off + i];
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
        if (c != suffix_lower[i])
            return false;
    }
    return true;
}

static bool is_image_path(const std::string &p)
{
    return ends_with_lower(p, ".jpg") || ends_with_lower(p, ".jpeg") || ends_with_lower(p, ".png") || ends_with_lower(p, ".bmp");
}

static void list_images_recursive(const std::string &dir_path, std::vector<std::string> &out)
{
    DIR *d = opendir(dir_path.c_str());
    if (d == NULL)
        return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        std::string p = dir_path;
        if (!p.empty() && p[p.size() - 1] != '/' && p[p.size() - 1] != '\\')
            p += "/";
        p += ent->d_name;

        struct stat st;
        if (stat(p.c_str(), &st) != 0)
            continue;

        if (S_ISDIR(st.st_mode))
        {
            list_images_recursive(p, out);
        }
        else
        {
            if (is_image_path(p))
                out.push_back(p);
        }
    }

    closedir(d);
}

static std::string basename_no_ext(const std::string &path)
{
    size_t p = path.find_last_of("/\\");
    std::string name = (p == std::string::npos) ? path : path.substr(p + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos)
        name = name.substr(0, dot);
    return name;
}

static std::string replace_images_to_labels(std::string path)
{
    std::string p = path;
    size_t pos = p.find("/images/");
    if (pos != std::string::npos)
    {
        p.replace(pos, strlen("/images/"), "/labels/");
        return p;
    }
    pos = p.find("\\images\\");
    if (pos != std::string::npos)
    {
        p.replace(pos, strlen("\\images\\"), "\\labels\\");
        return p;
    }
    return path;
}

static std::string label_path_from_image(const std::string &img_path, const std::string &label_dir)
{
    if (!label_dir.empty())
    {
        std::string base = basename_no_ext(img_path);
        return label_dir + "/" + base + ".txt";
    }

    std::string p = replace_images_to_labels(img_path);
    size_t dot = p.find_last_of('.');
    if (dot != std::string::npos)
        p = p.substr(0, dot);
    return p + ".txt";
}

static float iou(const BoxF &a, const BoxF &b)
{
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float iw = std::max(0.0f, inter_x2 - inter_x1);
    float ih = std::max(0.0f, inter_y2 - inter_y1);
    float inter = iw * ih;

    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    if (uni <= 0.0f)
        return 0.0f;
    return inter / uni;
}

static std::vector<BoxF> read_yolo_labels(const std::string &label_path, int img_w, int img_h)
{
    std::vector<BoxF> gts;
    std::ifstream f(label_path.c_str());
    if (!f.is_open())
        return gts;

    std::string line;
    while (std::getline(f, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        std::istringstream iss(line);
        float cid_f = 0.0f;
        float xc = 0.0f, yc = 0.0f, w = 0.0f, h = 0.0f;
        if (!(iss >> cid_f >> xc >> yc >> w >> h))
            continue;
        int cid = static_cast<int>(cid_f);
        if (cid != 0)
            continue;

        float x1 = (xc - w / 2.0f) * img_w;
        float y1 = (yc - h / 2.0f) * img_h;
        float x2 = (xc + w / 2.0f) * img_w;
        float y2 = (yc + h / 2.0f) * img_h;

        x1 = std::max(0.0f, std::min(x1, (float)(img_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(img_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(img_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(img_h - 1)));

        if (x2 <= x1 || y2 <= y1)
            continue;

        BoxF b;
        b.x1 = x1;
        b.y1 = y1;
        b.x2 = x2;
        b.y2 = y2;
        gts.push_back(b);
    }

    return gts;
}

struct Pred {
    int img_idx;
    BoxF box;
    float score;
};

struct Metrics {
    int tp;
    int fp;
    int fn;
    double precision;
    double recall;
    double f1;
    double ap;
    double best_precision;
    double best_recall;
    double best_f1;
    float best_conf;
};

static double ap_from_pr(const std::vector<double> &rec, const std::vector<double> &prec)
{
    std::vector<double> mrec;
    std::vector<double> mpre;
    mrec.reserve(rec.size() + 2);
    mpre.reserve(prec.size() + 2);

    mrec.push_back(0.0);
    mpre.push_back(0.0);
    for (size_t i = 0; i < rec.size(); i++)
    {
        mrec.push_back(rec[i]);
        mpre.push_back(prec[i]);
    }
    mrec.push_back(1.0);
    mpre.push_back(0.0);

    for (int i = (int)mpre.size() - 2; i >= 0; i--)
    {
        if (mpre[(size_t)i] < mpre[(size_t)i + 1])
            mpre[(size_t)i] = mpre[(size_t)i + 1];
    }

    double ap = 0.0;
    for (int t = 0; t <= 100; t++)
    {
        double r = (double)t / 100.0;
        size_t i = 0;
        while (i < mrec.size() && mrec[i] < r)
            i++;
        if (i >= mpre.size())
            i = mpre.size() - 1;
        ap += mpre[i];
    }
    ap /= 101.0;
    return ap;
}

static Metrics compute_metrics(const std::vector<std::vector<BoxF> > &all_gts,
                               const std::vector<Pred> &all_preds_sorted,
                               int total_gt,
                               float iou_th)
{
    Metrics m;
    m.tp = 0;
    m.fp = 0;
    m.fn = 0;
    m.precision = 0.0;
    m.recall = 0.0;
    m.f1 = 0.0;
    m.ap = 0.0;
    m.best_precision = 0.0;
    m.best_recall = 0.0;
    m.best_f1 = 0.0;
    m.best_conf = 0.0f;

    std::vector<std::vector<char> > gt_used;
    gt_used.resize(all_gts.size());
    for (size_t i = 0; i < all_gts.size(); i++)
    {
        gt_used[i].assign(all_gts[i].size(), 0);
    }

    std::vector<int> tp(all_preds_sorted.size(), 0);
    std::vector<int> fp(all_preds_sorted.size(), 0);

    for (size_t i = 0; i < all_preds_sorted.size(); i++)
    {
        const Pred &p = all_preds_sorted[i];
        if (p.img_idx < 0 || (size_t)p.img_idx >= all_gts.size())
        {
            m.fp++;
            fp[i] = 1;
            continue;
        }

        float best_iou = 0.0f;
        int best_j = -1;
        const std::vector<BoxF> &gts = all_gts[(size_t)p.img_idx];
        for (size_t j = 0; j < gts.size(); j++)
        {
            if (gt_used[(size_t)p.img_idx][j])
                continue;
            float v = iou(p.box, gts[j]);
            if (v > best_iou)
            {
                best_iou = v;
                best_j = (int)j;
            }
        }

        if (best_j >= 0 && best_iou >= iou_th)
        {
            gt_used[(size_t)p.img_idx][(size_t)best_j] = 1;
            m.tp++;
            tp[i] = 1;
        }
        else
        {
            m.fp++;
            fp[i] = 1;
        }
    }

    m.fn = total_gt - m.tp;
    if (m.fn < 0)
        m.fn = 0;

    if (m.tp + m.fp > 0)
        m.precision = (double)m.tp / (double)(m.tp + m.fp);
    if (total_gt > 0)
        m.recall = (double)m.tp / (double)total_gt;
    if (m.precision + m.recall > 0.0)
        m.f1 = 2.0 * m.precision * m.recall / (m.precision + m.recall);

    std::vector<double> prec_curve;
    std::vector<double> rec_curve;
    prec_curve.reserve(all_preds_sorted.size());
    rec_curve.reserve(all_preds_sorted.size());

    int cum_tp = 0;
    int cum_fp = 0;
    for (size_t i = 0; i < all_preds_sorted.size(); i++)
    {
        cum_tp += tp[i];
        cum_fp += fp[i];
        double pval = 0.0;
        double rval = 0.0;
        if (cum_tp + cum_fp > 0)
            pval = (double)cum_tp / (double)(cum_tp + cum_fp);
        if (total_gt > 0)
            rval = (double)cum_tp / (double)total_gt;
        prec_curve.push_back(pval);
        rec_curve.push_back(rval);

        double f1 = 0.0;
        if (pval + rval > 0.0)
            f1 = 2.0 * pval * rval / (pval + rval);
        if (f1 > m.best_f1)
        {
            m.best_f1 = f1;
            m.best_precision = pval;
            m.best_recall = rval;
            m.best_conf = all_preds_sorted[i].score;
        }
    }

    if (total_gt > 0 && !all_preds_sorted.empty())
        m.ap = ap_from_pr(rec_curve, prec_curve);

    return m;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("%s <model_path> <dataset_txt_or_images_dir> [label_dir] [iou_metric] [conf] [nms_iou]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *data_path = argv[2];
    std::string label_dir;
    if (argc >= 4)
        label_dir = argv[3];

    float iou_th = 0.5f;
    if (argc >= 5)
        iou_th = (float)atof(argv[4]);

    if (argc >= 6)
    {
        setenv("RKNN_BOX_THRESH", argv[5], 1);
    }
    else
    {
        const char *box_env = getenv("RKNN_BOX_THRESH");
        if (box_env == NULL || box_env[0] == '\0')
        {
            // Align with Ultralytics val default (very low conf to build PR curve)
            setenv("RKNN_BOX_THRESH", "0.001", 1);
        }
    }
    if (argc >= 7)
    {
        setenv("RKNN_NMS_THRESH", argv[6], 1);
    }
    else
    {
        const char *nms_env = getenv("RKNN_NMS_THRESH");
        if (nms_env == NULL || nms_env[0] == '\0')
        {
            // Align closer to Ultralytics NMS IoU default used in validation
            setenv("RKNN_NMS_THRESH", "0.7", 1);
        }
    }

    const char* box_env = getenv("RKNN_BOX_THRESH");
    const char* nms_env = getenv("RKNN_NMS_THRESH");
    printf("RKNN_BOX_THRESH=%s RKNN_NMS_THRESH=%s\n",
           (box_env && box_env[0]) ? box_env : "(default)",
           (nms_env && nms_env[0]) ? nms_env : "(default)");

    std::vector<std::string> img_paths;
    if (is_dir(data_path))
    {
        list_images_recursive(data_path, img_paths);
        std::sort(img_paths.begin(), img_paths.end());
    }
    else
    {
        std::ifstream list_f(data_path);
        if (!list_f.is_open())
        {
            printf("failed to open dataset_txt: %s\n", data_path);
            return -1;
        }

        std::string line;
        while (std::getline(list_f, line))
        {
            line = trim(line);
            if (!line.empty())
                img_paths.push_back(line);
        }
    }

    if (img_paths.empty())
    {
        printf("no images found: %s\n", data_path);
        return -1;
    }

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    std::vector<std::vector<BoxF> > all_gts;
    all_gts.reserve(img_paths.size());

    std::vector<Pred> all_preds;

    int total_gt = 0;
    int total_images = 0;

    for (size_t idx = 0; idx < img_paths.size(); idx++)
    {
        const std::string &img_path = img_paths[idx];

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = read_image(img_path.c_str(), &src_image);
        if (ret != 0)
        {
            all_gts.push_back(std::vector<BoxF>());
            continue;
        }

        total_images++;

        std::string lab_path = label_path_from_image(img_path, label_dir);
        std::vector<BoxF> gts = read_yolo_labels(lab_path, src_image.width, src_image.height);
        total_gt += (int)gts.size();
        all_gts.push_back(gts);

        object_detect_result_list od_results;
        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);

        if (src_image.virt_addr != NULL)
        {
            free(src_image.virt_addr);
            src_image.virt_addr = NULL;
        }

        if (ret != 0)
        {
            continue;
        }

        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det = &(od_results.results[i]);
            if (det->cls_id != 0)
                continue;
            Pred p;
            p.img_idx = (int)idx;
            p.score = det->prop;
            p.box.x1 = (float)det->box.left;
            p.box.y1 = (float)det->box.top;
            p.box.x2 = (float)det->box.right;
            p.box.y2 = (float)det->box.bottom;
            all_preds.push_back(p);
        }
    }

    std::sort(all_preds.begin(), all_preds.end(), [](const Pred &a, const Pred &b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.img_idx != b.img_idx)
            return a.img_idx < b.img_idx;
        if (a.box.x1 != b.box.x1)
            return a.box.x1 < b.box.x1;
        if (a.box.y1 != b.box.y1)
            return a.box.y1 < b.box.y1;
        if (a.box.x2 != b.box.x2)
            return a.box.x2 < b.box.x2;
        return a.box.y2 < b.box.y2;
    });

    Metrics base = compute_metrics(all_gts, all_preds, total_gt, iou_th);

    double ap50 = compute_metrics(all_gts, all_preds, total_gt, 0.50f).ap;
    double map5095 = 0.0;
    if (total_gt > 0 && !all_preds.empty())
    {
        double sum = 0.0;
        for (int k = 0; k < 10; k++)
        {
            float t = 0.50f + 0.05f * (float)k;
            sum += compute_metrics(all_gts, all_preds, total_gt, t).ap;
        }
        map5095 = sum / 10.0;
    }

    printf("images=%d gt=%d preds=%zu tp=%d fp=%d fn=%d\n", total_images, total_gt, all_preds.size(), base.tp, base.fp, base.fn);
    printf("Precision=%.6f Recall=%.6f F1=%.6f AP@%.2f=%.6f\n", base.precision, base.recall, base.f1, iou_th, base.ap);
    printf("BestF1@conf=%.6f: Precision=%.6f Recall=%.6f F1=%.6f\n", base.best_conf, base.best_precision, base.best_recall, base.best_f1);
    printf("AP@0.50=%.6f mAP@0.50:0.95=%.6f\n", ap50, map5095);

    release_yolo11_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
