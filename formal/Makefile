SBY ?= sby

.PHONY: all bmc cover clean

all: bmc cover

bmc:
	mkdir -p work
	$(SBY) -f -d work/tile_controller_banked_prefetch_bmc \
		tile_controller_banked_prefetch.sby bmc

cover:
	mkdir -p work
	$(SBY) -f -d work/tile_controller_banked_prefetch_cover \
		tile_controller_banked_prefetch.sby cover

clean:
	rm -rf work
