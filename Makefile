NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 
NVCCFLAGS = $(CXXFLAGS) -arch=sm_75 --compiler-options -Wall
INCLUDE_DIR = include
SRC_DIR = src
BUILD_DIR = build

CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%, $(CU_SOURCES))

all: $(BUILD_DIR) $(EXECUTABLES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

help:
	@echo "Available targets:"
	@echo "  all     - Build all benchmark executables"
	@echo "  clean   - Remove build directory"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Executables will be created in $(BUILD_DIR)/ directory"