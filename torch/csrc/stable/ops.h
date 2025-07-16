#pragma once

#include <torch/csrc/stable/library.h>
#include <array>
#include <cstdint>

using torch::stable::Tensor;

// We expect this to be the stable version of the tranpose op with identical
// semantics to the existing transpose.int op.
inline Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{};
  stack[0] = from(self);
  stack[1] = from(dim0);
  stack[2] = from(dim1);
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<Tensor>(stack[0]);
}
