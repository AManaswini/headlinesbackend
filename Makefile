#!/usr/bin/make -f
SHELL = /bin/sh
include make.config
include make.boiler
.DELETE_ON_ERROR:

# boilerplate variables, do not edit
MAKEFILE_PATH := $(abspath $(firstword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# required values, set to defaults here if not given in config.mk
PYTHON ?= python3.10
RSCRIPT ?= Rscript --quiet

# firm variables, usually the same but potentially require editing for current repo
CLEAN_DIRS_LIST = $(EXPERIMENT_DIR) $(PROJECT_DIR)
CLEAN_DIRS_REGEX = $(EXPERIMENT_DIR)|$(PROJECT_DIR)


# use regex to go through this Makefile and print help lines
# any comment starting with double '#' gets printed in help (see next comment for example).
# can add `| sort` to the `sed` command if you want to sort alphabetically, but better to write Makefile with a good order
## help :		print this help
.PHONY: help
help: Makefile make.boiler
	@echo "\nMake commands available:"
	@sed -n 's/^## /		/p' $^

# advice: we use `$<`, which equals the first dependency listed after the target. This (a) saves us from writing
# filepaths twice, and (b) forces us to include the script in the dependencies, which avoids accidentally excluding
# a file from the analysis process.
# note: we also begin each rule with a '@mkdir -p "${@D}"' command; this creates the directory that the output is 
# generated in automatically to avoid errors. However, scripts that produce multiple outputs may still fail unless the 
# other directories are added to the recipe.

## all :		make the entire project pipeline
all: data features regressions


# this block converts raw data into clean, analysis-ready data
## data :		convert raw data files into clean analysis datasets
.PHONY: data
data: data/clean/trial_randval.feather \
		data/clean/variations.feather \
		data/clean/headlines.feather \
		data/clean/trials.feather \
		data/clean/headline_ctr.feather

data/temp/variation_raw.feather: project/x01_data_cleaning/s01_combine_upworthy.R \
		data/raw/upworthy/upworthy-exploratory-03.12.2020.csv \
		data/raw/upworthy/upworthy-confirmatory-03.12.2020.csv \
		data/raw/upworthy/upworthy-holdout-03.12.2020.csv
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/trial_randval.feather: project/x01_data_cleaning/s02_create_upworthy_traintest_split.R \
		data/temp/variation_raw.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/variations.feather data/clean/headlines.feather data/clean/trials.feather : \
		project/x01_data_cleaning/s03_clean_upworthy.R \
		data/temp/variation_raw.feather \
		data/clean/trial_randval.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/temp/marginal_posterior_variables.feather data/temp/headline_variables.feather : \
		project/x01_data_cleaning/s04_sample_marginal_ctr_posterior.R \
		data/clean/headlines.feather \
		project/x01_data_cleaning/model_marginal_ctr.stan
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/headline_ctr.feather: project/x01_data_cleaning/s05_estimate_marginal_ctr_posterior.R \
		data/clean/trials.feather \
		data/clean/headlines.feather \
		data/temp/marginal_posterior_variables.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<


# this block runs the feature code
## features :	make the analysis datasets, with all necessary features
.PHONY: features
features: data \
		data/clean/B&U/variation_bu_filter.feather \
		data/clean/B&U/headline_bu_features.feather \
		data/clean/topics/headline_wordbags.feather \
		data/clean/embeddings/headline_embeddings_stsb_roberta_large.feather

data/clean/B&U/variation_bu_filter.feather: project/x02_features/s01_create_bu_filter.R \
		data/raw/upworthy/bandu-training_ids.xlsx \
		data/clean/variations.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/topics/liwc2022_results.feather data/clean/topics/liwc2015_results.feather data/clean/topics/textAnalyzer_results.feather : \
		project/x02_features/s02_create_bu_features_liwc_ta.R \
		data/raw/topics/liwc2022_results.csv \
		data/raw/topics/liwc2015_results.csv \
		data/raw/topics/textAnalyzer_results.csv
	@mkdir -p "${@D}"
	@echo "Now running a script that requires intervention--be wary!"
	$(RSCRIPT) $<

data/clean/topics/wordLists_results.feather: project/x02_features/s03_create_bu_features_constructs.R \
		data/clean/headlines.feather \
		data/raw/topics/banerjee_urminsky-psych_constructs.csv \
		data/raw/topics/banerjee_urminsky-topics.csv
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/B&U/headline_bu_features.feather: project/x02_features/s04_combine_bu_features.R \
		data/clean/headlines.feather \
		data/clean/topics/textAnalyzer_results.feather \
		data/clean/topics/liwc2022_results.feather \
		data/clean/topics/liwc2015_results.feather \
		data/clean/topics/wordLists_results.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/topics/headline_wordbags.feather: project/x02_features/s05_create_wordbag_features.R \
		data/clean/headlines.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/embeddings/headline_embeddings_stsb_roberta_large.feather: project/x02_features/s06_extract_embedding_features.py \
		data/clean/headlines.feather
	@mkdir -p "${@D}"
	$(PYTHON) $<


## regressions :	run all analysis regressions
.PHONY: regressions
regressions: data \
		features \
		output/regression/bu_recreation_results.csv \
		data/clean/predictions/headline_predictions-embedding_ridge.feather \
		output/regression/benchmark_models_results.csv

output/regression/bu_recreation_results.csv: project/x03_regressions/s01_recreate_bu_results.R \
		data/clean/variations.feather \
		data/clean/B&U/variation_bu_filter.feather \
		data/clean/B&U/headline_bu_features.feather \
		data/clean/embeddings/headline_embeddings_stsb_roberta_large.feather \
		data/clean/topics/headline_wordbags.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

data/clean/predictions/headline_predictions-embedding_ridge.feather: project/x03_regressions/s02_fit_embedding_model.R \
		data/clean/headlines.feather \
		data/clean/headline_ctr.feather \
		data/clean/embeddings/headline_embeddings_stsb_roberta_large.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<

output/regression/benchmark_models_results.csv: project/x03_regressions/s03_fit_benchmark_models.R \
		data/clean/headlines.feather \
		data/clean/headline_ctr.feather \
		data/clean/B&U/headline_bu_features.feather \
		data/clean/topics/headline_wordbags.feather \
		data/clean/predictions/headline_predictions-embedding_ridge.feather \
		data/clean/embeddings/headline_embeddings_stsb_roberta_large.feather \
		data/clean/prolific_humanGuesses/qualtrics_1d/headline_pairs-sample-humanGuesses-1d.feather
	@mkdir -p "${@D}"
	$(RSCRIPT) $<
