MODEL_NAME       ?= simple
EPOCHS           ?= 10
BATCH_SIZE       ?= 4
LR               ?= 1e-3
INPUT_FRAMES     ?= 20
OUTPUT_FRAMES    ?= 5
IMG_SIZE         ?= 128
DEVICE           ?= cuda

INPUT_DIM        ?= 1
HIDDEN_DIM       ?= 32
N_LAYERS         ?= 2
KERNEL_SIZE      ?= 3
PADDING          ?= 1
ATT_HIDDEN_DIM   ?= 16

.PHONY: run
run:
	@echo "Running model: $(MODEL_NAME)"
	python main.py \
		--model_name $(MODEL_NAME) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--lr $(LR) \
		--input_frames $(INPUT_FRAMES) \
		--output_frames $(OUTPUT_FRAMES) \
		--img_size $(IMG_SIZE) \
		--device $(DEVICE) \
		--input_dim $(INPUT_DIM) \
		--hidden_dim $(HIDDEN_DIM) \
		--n_layers $(N_LAYERS) \
		--kernel_size $(KERNEL_SIZE) \
		--padding $(PADDING) \
		--att_hidden_dim $(ATT_HIDDEN_DIM)

.PHONY: simple
simple:
	$(MAKE) run MODEL_NAME=simple

.PHONY: encode2decode
encode2decode:
	$(MAKE) run MODEL_NAME=encode2decode

.PHONY: encode2decode_unet
encode2decode_unet:
	$(MAKE) run MODEL_NAME=encode2decode_unet

.PHONY: sa_encode2decode
sa_encode2decode:
	$(MAKE) run MODEL_NAME=sa_encode2decode

.PHONY: sa_encode2decode_gan
sa_encode2decode_gan:
	$(MAKE) run MODEL_NAME=sa_encode2decode_gan

.PHONY: help
help:
	@echo "Makefile usage:"
	@echo "  make [run|simple|encode2decode|encode2decode_unet|sa_encode2decode|sa_encode2decode_gan|gan_long|... ]"
	@echo
	@echo "Variables you can override:"
	@echo "  MODEL_NAME (default=simple)"
	@echo "  EPOCHS (default=10)"
	@echo "  BATCH_SIZE (default=4)"
	@echo "  LR (default=1e-3)"
	@echo "  INPUT_FRAMES (default=5)"
	@echo "  OUTPUT_FRAMES (default=5)"
	@echo "  IMG_SIZE (default=128)"
	@echo "  DEVICE (default=cuda)"
	@echo "  INPUT_DIM (default=1)"
	@echo "  HIDDEN_DIM (default=32)"
	@echo "  N_LAYERS (default=2)"
	@echo "  KERNEL_SIZE (default=3)"
	@echo "  PADDING (default=1)"
	@echo "  ATT_HIDDEN_DIM (default=16)"
	@echo
	@echo "Example to run 'sa_encode2decode' for 15 epochs with batch=8 :"
	@echo "   make sa_encode2decode EPOCHS=15 BATCH_SIZE=8"
	@echo "Example to run custom config with run target :"
	@echo "   make run MODEL_NAME=simple EPOCHS=20 LR=1e-4"