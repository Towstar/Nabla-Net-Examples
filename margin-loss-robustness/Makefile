CMAKE ?= cmake
BUILD_DIR ?= build
CONFIG ?= Release
TARGET := nablanet_softmax_classification

.PHONY: configure build clean

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(CONFIG) --target $(TARGET)

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
