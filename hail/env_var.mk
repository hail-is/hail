# $(1) is an environment variable name
#
# Example:
#
#    VERSION ?= 30
#    $(eval $(call ENV_VAR,VERSION))
#
#    build: env/VERSION
#    build:
#      ...
#
# If VERSION is set on the command line: `VERSION=31 make` and make was
# previously called with VERSION set to a different value, then `build` will be
# marked out-of-date.

define ENV_VAR
ifneq ($$($(1)),$$(shell cat env/$(1) 2>/dev/null))
.PHONY: env/$(1)
env/$(1):
	$$(info $(1) is set to "$$($(1))" which is different from old value "$$(shell cat env/$(1) 2>/dev/null)")
	@mkdir -p env
	printf "$$($(1))" > $$@
else ifeq (,$$(wildcard env/$(1)))
.PHONY: env/$(1)
env/$(1):
	$$(info creating env/$(1) which does not exist)
	@mkdir -p env
	@touch $$@
endif
endef

.PHONY: clean-env
clean-env:
	rm -rf env
