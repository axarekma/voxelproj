#include <torch/extension.h>
#include <vector>

// Updated declarations with optional block size
void forward_z0(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8});
void backward_z0(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8});
void forward_z2(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8});
void backward_z2(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8});
void backward_z0_opt(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8});

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.doc() = "Projection Kernels.";

      m.def("forward_z0", &forward_z0,
            py::arg("in"), py::arg("out"), py::arg("angles"),
            py::arg("block_size") = std::vector<int>{8, 8, 8},
            "Forward projection. Height as the fastest index.");

      m.def("backward_z0", &backward_z0,
            py::arg("in"), py::arg("out"), py::arg("angles"),
            py::arg("block_size") = std::vector<int>{8, 8, 8},
            "Backward projection. Height as the fastest index.");

      m.def("forward_z2", &forward_z2,
            py::arg("in"), py::arg("out"), py::arg("angles"),
            py::arg("block_size") = std::vector<int>{8, 8, 8},
            "Forward projection. Height as the slowest index.");

      m.def("backward_z2", &backward_z2,
            py::arg("in"), py::arg("out"), py::arg("angles"),
            py::arg("block_size") = std::vector<int>{8, 8, 8},
            "Backward projection. Height as the slowest index.");
}
