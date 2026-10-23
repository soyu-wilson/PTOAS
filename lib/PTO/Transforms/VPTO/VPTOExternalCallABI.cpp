// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Native external-call ABI preparation and fallback metadata for the split
// VPTO emitter. Ordinary kernels bypass descriptor preparation entirely.

#include "VPTOCANN900LLVMEmitterInternal.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::pto::detail {
namespace {

constexpr llvm::StringLiteral kPreserveExternalCallMemRefsAttr =
    "pto.preserve_external_call_memrefs";
constexpr llvm::StringLiteral kNativeTemplateFallbackSymbolsAttr =
    "pto.native_template_fallback_symbols";

Type convertExternalMemRefAddressSpace(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) {
    return type;
  }
  auto addressSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!addressSpace) {
    return type;
  }
  auto integerSpace = IntegerAttr::get(IntegerType::get(type.getContext(), kBits64),
                                      static_cast<int64_t>(addressSpace.getAddressSpace()));
  return MemRefType::Builder(memrefType).setMemorySpace(integerSpace);
}

void normalizeExternalCallTypes(Operation *op) {
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    FunctionType oldType = func.getFunctionType();
    auto inputs = llvm::map_to_vector(oldType.getInputs(), convertExternalMemRefAddressSpace);
    auto results = llvm::map_to_vector(oldType.getResults(), convertExternalMemRefAddressSpace);
    func.setFunctionType(FunctionType::get(op->getContext(), inputs, results));
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument argument : block.getArguments()) {
        argument.setType(convertExternalMemRefAddressSpace(argument.getType()));
      }
    }
  }
  for (OpResult result : op->getResults()) {
    result.setType(convertExternalMemRefAddressSpace(result.getType()));
  }
}

LogicalResult validateCarrierMetadata(memref::ExtractStridedMetadataOp metadata) {
  for (OpResult result : metadata->getResults().drop_front()) {
    if (!result.use_empty()) {
      return metadata.emitError("opaque external-call carrier metadata must be unused");
    }
  }
  Value baseBuffer = metadata->getResult(0);
  for (Operation *user : baseBuffer.getUsers()) {
    auto view = dyn_cast<memref::ReinterpretCastOp>(user);
    const bool validView = view && view.getSource() == baseBuffer;
    if (!validView) {
      return metadata.emitError("opaque external-call carrier base must feed only memref.reinterpret_cast views");
    }
  }
  return success();
}

LogicalResult validateDynamicCarrier(UnrealizedConversionCastOp bridge, MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  const bool validLayout = type.getRank() == 1 && succeeded(getStridesAndOffset(type, strides, offset)) &&
      strides.size() == 1 && strides.front() == 1 && offset == 0;
  if (!validLayout) {
    return bridge.emitError("dynamic external-call pointer bridge requires a rank-1 identity-layout carrier memref");
  }
  for (Operation *user : bridge.getResult(0).getUsers()) {
    auto metadata = dyn_cast<memref::ExtractStridedMetadataOp>(user);
    const bool validMetadata = metadata && metadata.getSource() == bridge.getResult(0);
    if (!validMetadata) {
      return bridge.emitError("dynamic external-call pointer bridge must feed only memref.extract_strided_metadata");
    }
    if (failed(validateCarrierMetadata(metadata))) {
      return failure();
    }
  }
  return success();
}

LogicalResult materializeDynamicCarrier(UnrealizedConversionCastOp bridge, MemRefType type,
                                       LLVMTypeConverter &converter) {
  if (failed(validateDynamicCarrier(bridge, type))) {
    return failure();
  }
  Type descriptorType = converter.convertType(type);
  if (!descriptorType) {
    return bridge.emitError("failed to convert dynamic carrier memref type");
  }
  OpBuilder builder(bridge);
  MemRefDescriptor descriptor = MemRefDescriptor::undef(builder, bridge.getLoc(), descriptorType);
  descriptor.setAllocatedPtr(builder, bridge.getLoc(), bridge.getOperand(0));
  descriptor.setAlignedPtr(builder, bridge.getLoc(), bridge.getOperand(0));
  descriptor.setConstantOffset(builder, bridge.getLoc(), 0);
  descriptor.setConstantSize(builder, bridge.getLoc(), 0, 0);
  descriptor.setConstantStride(builder, bridge.getLoc(), 0, 1);
  bridge->setOperand(0, descriptor);
  return success();
}

LogicalResult materializePointerBridge(UnrealizedConversionCastOp bridge, LLVMTypeConverter &converter) {
  auto type = cast<MemRefType>(bridge.getResult(0).getType());
  if (!type.hasStaticShape()) {
    return materializeDynamicCarrier(bridge, type, converter);
  }
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  const bool staticLayout = succeeded(getStridesAndOffset(type, strides, offset)) &&
      !llvm::is_contained(strides, ShapedType::kDynamic) && offset != ShapedType::kDynamic;
  if (!staticLayout) {
    return bridge.emitError("external-call pointer bridge requires a static strided memref");
  }
  OpBuilder builder(bridge);
  MemRefDescriptor descriptor = MemRefDescriptor::fromStaticShape(
      builder, bridge.getLoc(), converter, type, bridge.getOperand(0));
  bridge->setOperand(0, descriptor);
  return success();
}

class PrepareExternalCallMemRefABIPass final
    : public PassWrapper<PrepareExternalCallMemRefABIPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareExternalCallMemRefABIPass)

  StringRef getArgument() const final {
    return "pto-prepare-external-call-memref-abi-cann900";
  }

  StringRef getDescription() const final {
    return "Prepare preserved external-call memrefs for standard LLVM ABI lowering";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr(kPreserveExternalCallMemRefsAttr)) {
      return;
    }
    module.walk(normalizeExternalCallTypes);
    LowerToLLVMOptions options(module.getContext());
    LLVMTypeConverter converter(module.getContext(), options);
    SmallVector<UnrealizedConversionCastOp> bridges;
    module.walk([&](UnrealizedConversionCastOp bridge) {
      const bool pointerBridge = bridge->getNumOperands() == 1 && bridge->getNumResults() == 1 &&
          isa<LLVM::LLVMPointerType>(bridge.getOperand(0).getType()) &&
          isa<MemRefType>(bridge.getResult(0).getType());
      if (pointerBridge) {
        bridges.push_back(bridge);
      }
    });
    for (UnrealizedConversionCastOp bridge : bridges) {
      if (failed(materializePointerBridge(bridge, converter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPrepareExternalCallMemRefABIPass() {
  return std::make_unique<PrepareExternalCallMemRefABIPass>();
}

void attachNativeTemplateFallbackMetadata(ModuleOp sourceModule, llvm::Module &llvmModule) {
  auto symbols = sourceModule->getAttrOfType<ArrayAttr>(kNativeTemplateFallbackSymbolsAttr);
  if (!symbols) {
    return;
  }
  llvm::NamedMDNode *metadata = llvmModule.getOrInsertNamedMetadata(kNativeTemplateFallbackSymbolsAttr);
  for (StringAttr symbol : symbols.getAsRange<StringAttr>()) {
    metadata->addOperand(llvm::MDNode::get(llvmModule.getContext(),
        llvm::MDString::get(llvmModule.getContext(), symbol.getValue())));
  }
}

} // namespace mlir::pto::detail
