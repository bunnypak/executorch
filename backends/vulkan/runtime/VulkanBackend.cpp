/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/VulkanDelegateHeader.h>
#include <executorch/backends/vulkan/schema_generated.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/profiler.h>

#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>
#include <type_traits>
#include <vector>

namespace torch {
namespace executor {
namespace vulkan {
namespace {

using namespace at::native::vulkan;

// Flatbuffer types
using VkGraphPtr = const vkgraph::VkGraph*;
using OpCallPtr = const vkgraph::OperatorCall*;
using VkValuePtr = const vkgraph::VkValue*;
using VkTensorPtr = const vkgraph::VkTensor*;
using VkBytesPtr = const vkgraph::VkBytes*;

// Flatbuffer vector types
using VkValuesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkValue>>*;
using BytesVector =
    const flatbuffers::Vector<flatbuffers::Offset<vkgraph::VkBytes>>*;
using UIntVector = const flatbuffers::Vector<uint32_t>*;

const uint8_t* getConstantDataPtr(
    VkGraphPtr flatbuffer_graph,
    const int32_t buffer_idx,
    const uint8_t* constant_data) {
  VkBytesPtr constant_bytes = flatbuffer_graph->constants()->Get(buffer_idx);
  return constant_data + constant_bytes->offset();
}

api::ScalarType get_scalar_type(const vkgraph::VkDataType& vk_datatype) {
  switch (vk_datatype) {
    case (vkgraph::VkDataType::fp32): {
      return api::kFloat;
    }
  }
}

class GraphBuilder {
  ComputeGraph* compute_graph_;
  VkGraphPtr flatbuffer_;
  const uint8_t* constant_data_;

  std::unordered_map<uint32_t, ValueRef> ref_mapping_;

 public:
  explicit GraphBuilder(
      ComputeGraph* compute_graph,
      VkGraphPtr flatbuffer,
      const uint8_t* constant_data)
      : compute_graph_(compute_graph),
        flatbuffer_(flatbuffer),
        constant_data_(constant_data),
        ref_mapping_() {}

  bool fb_id_exists(const uint32_t fb_id) {
    const std::unordered_map<uint32_t, ValueRef>::iterator found_ref =
        ref_mapping_.find(fb_id);

    return found_ref != ref_mapping_.end();
  }

  ValueRef get_fb_id_valueref(const uint32_t fb_id) {
    const std::unordered_map<uint32_t, ValueRef>::iterator found_ref =
        ref_mapping_.find(fb_id);

    ET_CHECK_MSG(
        found_ref != ref_mapping_.end(),
        "Trying to extract a value that hasn't yet been added to the graph.");

    return found_ref->second;
  }

  void add_tensor_to_graph(const uint32_t fb_id, VkTensorPtr tensor_fb) {
    const api::ScalarType& dtype = get_scalar_type(tensor_fb->datatype());

    UIntVector dims_fb = tensor_fb->dims();
    const std::vector<int64_t> dims_vector(dims_fb->cbegin(), dims_fb->cend());

    ValueRef ref;
    if (tensor_fb->constant_id() >= 0) {
      const uint8_t* tensor_data = getConstantDataPtr(
          flatbuffer_, tensor_fb->constant_id(), constant_data_);

      ref = compute_graph_->add_tensorref(dims_vector, dtype, tensor_data);
    } else {
      ref = compute_graph_->add_tensor(
          dims_vector, dtype, tensor_fb->mem_obj_id());
    }

    ref_mapping_[fb_id] = ref;
  }

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, void>::type
  add_scalar_to_graph(const uint32_t fb_id, T value) {
    ValueRef ref = compute_graph_->add_scalar(value);
    ref_mapping_[fb_id] = ref;
  }

  void add_string_to_graph(const uint32_t fb_id, VkValuePtr value) {
    const auto fb_str = value->value_as_String()->string_val();
    std::string string(fb_str->cbegin(), fb_str->cend());
    ValueRef ref = compute_graph_->add_string(std::move(string));
    ref_mapping_[fb_id] = ref;
  }

  void add_value_to_graph(const uint32_t fb_id, VkValuePtr value) {
    ET_CHECK_MSG(
        !fb_id_exists(fb_id),
        "Trying to add a value that has already been added to the graph.");

    switch (value->value_type()) {
      case vkgraph::GraphTypes::Int:
        add_scalar_to_graph(fb_id, value->value_as_Int()->int_val());
        break;
      case vkgraph::GraphTypes::Double:
        add_scalar_to_graph(fb_id, value->value_as_Double()->double_val());
        break;
      case vkgraph::GraphTypes::Bool:
        add_scalar_to_graph(fb_id, value->value_as_Bool()->bool_val());
        break;
      case vkgraph::GraphTypes::VkTensor:
        add_tensor_to_graph(fb_id, value->value_as_VkTensor());
        break;
      case vkgraph::GraphTypes::String:
        add_string_to_graph(fb_id, value);
        break;
      default:
        ET_CHECK_MSG(false, "Unsupported value type.");
    }
  }

  void build_graph() {
    // First, add all values to the graph
    for (uint32_t fb_id = 0; fb_id < flatbuffer_->values()->size(); ++fb_id) {
      VkValuePtr value = flatbuffer_->values()->Get(fb_id);
      add_value_to_graph(fb_id, value);
    }

    // Parse the inputs
    for (const uint32_t fb_id : *flatbuffer_->input_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      compute_graph_->set_input_tensor(ref);
    }

    // Parse the operators
    for (OpCallPtr op_call : *(flatbuffer_->chain())) {
      std::string op_name = op_call->name()->str();
      ET_CHECK_MSG(VK_HAS_OP(op_name), "Missing operator: %s", op_name.c_str());

      const std::vector<int> arg_fb_ids(
          op_call->args()->cbegin(), op_call->args()->cend());

      std::vector<ValueRef> args;
      for (const int arg_fb_id : arg_fb_ids) {
        args.push_back(get_fb_id_valueref(arg_fb_id));
      }

      auto vkFn = VK_GET_OP_FN(op_name);
      vkFn(*compute_graph_, args);
    }

    // Parse the outputs
    for (const uint32_t fb_id : *flatbuffer_->output_ids()) {
      const ValueRef ref = get_fb_id_valueref(fb_id);
      compute_graph_->set_output_tensor(ref);
    }
  }
};

class VulkanBackend final : public PyTorchBackendInterface {
 public:
  ~VulkanBackend() override = default;

  bool is_available() const override {
    // TODO(ssjia): replace with an actual Vulkan runtime availability check
    return true;
  }

  __ET_NODISCARD Error
  compileModel(const void* buffer_pointer, ComputeGraph* compute_graph) const {
    Result<VulkanDelegateHeader> header =
        VulkanDelegateHeader::Parse(buffer_pointer);

    const uint8_t* flatbuffer_data = nullptr;
    const uint8_t* constant_data = nullptr;

    if (header.ok()) {
      const uint8_t* buffer_start =
          reinterpret_cast<const uint8_t*>(buffer_pointer);
      flatbuffer_data = buffer_start + header->flatbuffer_offset;
      constant_data = buffer_start + header->bytes_offset;
    } else {
      ET_LOG(Error, "VulkanDelegateHeader may be corrupt");
      return header.error();
    }

    ET_CHECK_OR_RETURN_ERROR(
        vkgraph::VkGraphBufferHasIdentifier(flatbuffer_data),
        DelegateInvalidCompatibility,
        "Vulkan Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(flatbuffer_data),
        vkgraph::VkGraphIdentifier());

    VkGraphPtr flatbuffer_graph = vkgraph::GetVkGraph(flatbuffer_data);

    GraphBuilder builder =
        GraphBuilder(compute_graph, flatbuffer_graph, constant_data);

    builder.build_graph();

    compute_graph->prepare();

    compute_graph->encode_prepack();
    compute_graph->prepack();

    compute_graph->encode_execute();

    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec>) const override {
    ComputeGraph* compute_graph = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), ComputeGraph);

    new (compute_graph) ComputeGraph(GraphConfig());

    Error err = compileModel(processed->data(), compute_graph);

    if (err != Error::Ok) {
      return err;
    }

    return compute_graph;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    EXECUTORCH_SCOPE_PROF("VulkanBackend::execute");

    ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);

    const size_t num_inputs = compute_graph->inputs().size();
    for (size_t i = 0; i < num_inputs; i++) {
      compute_graph->copy_into_staging(
          compute_graph->inputs()[i],
          args[i]->toTensor().const_data_ptr(),
          args[i]->toTensor().numel());
    }

    compute_graph->execute();

    for (size_t i = 0; i < compute_graph->outputs().size(); i++) {
      // args holds inputs directly followed by outputs, so the i'th output
      // for compute_graph corresponds to the (i + num_inputs)'th arg
      compute_graph->copy_from_staging(
          compute_graph->outputs()[i],
          args[num_inputs + i]->toTensor().mutable_data_ptr(),
          args[num_inputs + i]->toTensor().numel());
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      ComputeGraph* compute_graph = static_cast<ComputeGraph*>(handle);
      // ComputeGraph is not trivially destructible. Since
      // this was constructed manually in init(), we must destroy it manually
      // here.
      compute_graph->~ComputeGraph();
    }
  }
};

auto cls = VulkanBackend();
Backend backend{"VulkanBackend", &cls};
static auto success_with_compiler = register_backend(backend);

} // namespace
} // namespace vulkan
} // namespace executor
} // namespace torch
