/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorShapeDynamism;
using torch::executor::testing::TensorFactory;

Tensor& op_cos_out(const Tensor& self, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::cos_outf(context, self, out);
}

TEST(OpCosOutKernelTest, HandleBoolInput) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;

  const std::vector<int32_t> sizes = {1, 2};

  Tensor a = tf_bool.make(sizes, /*data=*/{false, true});
  Tensor out = tf_float.zeros(sizes);
  Tensor res = tf_float.make(sizes, /*data=*/{1.000000, 0.540302});

  EXPECT_TENSOR_CLOSE(op_cos_out(a, out), res);
}

// Common testing for cos operator and all kinds of supported input types
template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
void test_floating_point_cos_out(
    const std::vector<int32_t>& out_shape = {1, 6},
    TensorShapeDynamism dynamism = TensorShapeDynamism::STATIC) {
  TensorFactory<IN_DTYPE> tf_in;
  TensorFactory<OUT_DTYPE> tf_out;

  // Destination for the cos operator.
  Tensor out = tf_out.zeros(out_shape, dynamism);

  // clang-format off
  op_cos_out(tf_in.make({1, 6}, { 0, 1, 3, 5, 10, 100 }), out);

  // Check that it matches (or close to) the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make({1, 6}, { 1.000000,  0.540302, -0.989992,  0.283662, -0.839072,  0.862319 }));
  // clang-format on
}

TEST(OpCosOutKernelTest, AllRealInputHalfOutputStaticDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputFloatOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Float>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputDoubleOutputStaticDynamismSupport) {
#define TEST_ENTRY(ctype, dtype) \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputHalfOutputBoundDynamismSupport) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Test Half support only for ExecuTorch mode";
  }
#define TEST_ENTRY(ctype, dtype)                                     \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Float>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REALH_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputFloatOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                     \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Float>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputDoubleOutputBoundDynamismSupport) {
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Double>( \
      {10, 10}, TensorShapeDynamism::DYNAMIC_BOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputFloatOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                     \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Float>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST(OpCosOutKernelTest, AllRealInputDoubleOutputUnboundDynamismSupport) {
  if (!torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
#define TEST_ENTRY(ctype, dtype)                                      \
  test_floating_point_cos_out<ScalarType::dtype, ScalarType::Double>( \
      {1, 1}, TensorShapeDynamism::DYNAMIC_UNBOUND);
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Unhandled output dtypes.
template <ScalarType INPUT_DTYPE, ScalarType OUTPUT_DTYPE>
void test_cos_invalid_output_dtype_dies() {
  TensorFactory<INPUT_DTYPE> tf;
  TensorFactory<OUTPUT_DTYPE> tf_out;

  const std::vector<int32_t> sizes = {2, 5};

  Tensor in = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(op_cos_out(in, out));
}

TEST(OpCosOutKernelTest, AllNonFloatOutputDTypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_cos_invalid_output_dtype_dies<ScalarType::Float, ScalarType::dtype>();
  ET_FORALL_INT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

// Mismatched shape tests.
TEST(OpCosOutKernelTest, MismatchedInputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input shapes";
  }
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4});
  Tensor out = tf.ones(/*sizes=*/{2, 2});

  ET_EXPECT_KERNEL_FAILURE(op_cos_out(a, out));
}
