# Thin wrapper over build.sh. Override: make ARCH=100a NCCL_HOME=/path TAG=x
ARCH ?= 107a
TAG ?= $(shell date +%m%d-%H%M)
export ARCH NVCC NCCL_HOME NCCL_INC NCCL_LIB FP4AR_DIAG BUILD_DIR
.PHONY: all check verify clean
all:
	./build.sh all $(TAG)
check:
	./build.sh check $(TAG)
verify:
	./build.sh verify
clean:
	rm -rf build
