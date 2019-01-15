# $(1) is an environment variable name

define ENV_VAR
ifneq ($$($(1)),$$(shell cat env/$(1)))
$$(info $(1) is set to "$$($(1))" which is different from old value "$$(shell cat env/$(1))")
.PHONY: env/$(1)
env/$(1):
	mkdir -p env
	printf "$$($(1))" > $$@
endif
endef

.PHONY: clean-env
clean-env:
	rm -rf env
