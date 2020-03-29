#include <torch/extension.h>
#include <pybind11/pybind11.h>

at::Tensor nms_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);

at::Tensor nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);


at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
    if (dets.device().is_cuda()) {
        return nms_cuda(dets, scores, iou_threshold);
    }
    return nms_cpu(dets, scores, iou_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Docstring is here!"; // optional module docstring
    m.def("nms", &nms, "Function instruction is here!");
}