# @file NewDataConversion.R
#
# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of cyclops
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

isSorted <- function(data, columnNames, ascending = rep(TRUE, length(columnNames))){
    return(.isSorted(data, columnNames, ascending))
}

#' Convert data from two data frames or ffdf objects into a CyclopsData object
#'
#' @description
#' \code{convertToCyclopsData} loads data from two data frames or ffdf objects, and inserts it into a Cyclops data object.
#'
#' @param outcomes      A data frame or ffdf object containing the outcomes with predefined columns (see below).
#' @param covariates    A data frame or ffdf object containing the covariates with predefined columns (see below).
#' @param modelType     Cyclops model type. Current supported types are "pr", "cpr", lr", "clr", or "cox"
#' @param addIntercept  Add an intercept to the model?
#' @param checkSorting  (DEPRECATED) Check if the data are sorted appropriately, and if not, sort.
#' @param checkRowIds   Check if all rowIds in the covariates appear in the outcomes.
#' @param normalize     String: Name of normalization for all non-indicator covariates (possible values: stdev, max, median)
#' @param quiet         If true, (warning) messages are suppressed.
#' @param floatingPoint Specified floating-point representation size (32 or 64)
#' @param timeEffectMap A data frame or ffdf object containing the convariates that have time-varying effects on the outcome
#'
#' @details
#' These columns are expected in the outcome object:
#' \tabular{lll}{
#'   \verb{stratumId}    \tab(integer) \tab (optional) Stratum ID for conditional regression models \cr
#'   \verb{rowId}  	\tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{y}    \tab(real) \tab The outcome variable \cr
#'   \verb{time}    \tab(real) \tab For models that use time (e.g. Poisson or Cox regression) this contains time \cr
#'                  \tab        \tab(e.g. number of days) \cr
#'   \verb{weights} \tab(real) \tab (optional) Non-negative weights to apply to outcome \cr
#'   \verb{censorWeights} \tab(real) \tab (optional) Non-negative censoring weights for competing risk model; will be computed if not provided.
#' }
#'
#' These columns are expected in the covariates object:
#' \tabular{lll}{
#'   \verb{stratumId}    \tab(integer) \tab (optional) Stratum ID for conditional regression models \cr
#'   \verb{rowId}  	\tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{covariateId}    \tab(integer) \tab A numeric identifier of a covariate  \cr
#'   \verb{covariateValue}    \tab(real) \tab The value of the specified covariate \cr
#' }
#'
#' These columns are expected in the timeEffectMap object:
#' \tabular{lll}{
#'   \verb{covariateId}    \tab(integer) \tab A numeric identifier of the covariates that have time-varying effects on the outcome \cr
#' }
#'
#' @return
#' An object of type cyclopsData
#'
#' @examples
#' #Convert infert dataset to Cyclops format:
#' covariates <- data.frame(stratumId = rep(infert$stratum, 2),
#'                          rowId = rep(1:nrow(infert), 2),
#'                          covariateId = rep(1:2, each = nrow(infert)),
#'                          covariateValue = c(infert$spontaneous, infert$induced))
#' outcomes <- data.frame(stratumId = infert$stratum,
#'                        rowId = 1:nrow(infert),
#'                        y = infert$case)
#' #Make sparse:
#' covariates <- covariates[covariates$covariateValue != 0, ]
#'
#' #Create Cyclops data object:
#' cyclopsData <- convertToCyclopsData(outcomes, covariates, modelType = "clr",
#'                                     addIntercept = FALSE)
#'
#' #Fit model:
#' fit <- fitCyclopsModel(cyclopsData, prior = createPrior("none"))
#'
#' @export
convertToCyclopsData <- function(outcomes,
                                 covariates,
                                 modelType = "lr",
				 timeEffectMap = NULL,
                                 addIntercept = TRUE,
                                 checkSorting = NULL,
                                 checkRowIds = TRUE,
                                 normalize = NULL,
                                 quiet = FALSE,
                                 floatingPoint = 64) {
    UseMethod("convertToCyclopsData")
}

#' @describeIn convertToCyclopsData Convert data from two \code{data.frame}
#' @export
# <<<<<<< HEAD
# convertToCyclopsData.ffdf <- function(outcomes,
#                                       covariates,
#                                       modelType = "lr",
#                                       addIntercept = TRUE,
#                                       checkSorting = TRUE,
#                                       checkRowIds = TRUE,
#                                       normalize = NULL,
#                                       quiet = FALSE,
#                                       floatingPoint = 64){
#     if ((modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron") & addIntercept){
#         if(!quiet) {
#             warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
#         }
# =======
convertToCyclopsData.data.frame <- function(outcomes,
                                            covariates,
                                            modelType = "lr",
					    timeEffectMap = NULL,
                                            addIntercept = TRUE,
                                            checkSorting = NULL,
                                            checkRowIds = TRUE,
                                            normalize = NULL,
                                            quiet = FALSE,
                                            floatingPoint = 64) {
    if (!is.null(checkSorting))
        warning("The 'checkSorting' argument has been deprecated. Sorting is now always checked")

    if ((modelType == "clr" | modelType == "cpr") & addIntercept) {
        if (!quiet)
            warning("Intercepts are not allowed in conditional models, removing intercept", call. = FALSE)
# >>>>>>> master
        addIntercept = FALSE
    }
    if (modelType == "pr" | modelType == "cpr")
        if (any(outcomes$time <= 0))
            stop("time cannot be non-positive", call. = FALSE)

    if (modelType == "lr" | modelType == "pr") {
        outcomes$stratumId <- NULL
        covariates$stratumId <- NULL
    }
    if ((modelType == "cox" | modelType == "cox_time" | modelType == "fgr") & !"stratumId" %in% colnames(outcomes)) {
        outcomes$stratumId <- 0
        covariates$stratumId <- 0
    }

    if (modelType == "lr" | modelType == "pr") {
        if (!isSorted(outcomes, c("rowId"))) {
            if (!quiet)
                writeLines("Sorting outcomes by rowId")
            outcomes <- outcomes[order(outcomes$rowId),]
        }
        if (!isSorted(covariates, c("covariateId", "rowId"))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId and rowId")
            covariates <- covariates[order(covariates$covariateId, covariates$rowId),]
        }
    }

    if (modelType == "clr" | modelType == "cpr") {
        if (!isSorted(outcomes, c("stratumId","rowId"))) {
            if (!quiet)
                writeLines("Sorting outcomes by stratumId and rowId")
            outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
        }
        if (!isSorted(covariates, c("covariateId", "stratumId","rowId"))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId, stratumId, and rowId")
            covariates <- covariates[order(covariates$covariateId, covariates$stratumId, covariates$rowId),]
        }
    }

    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr") {

        if ((modelType == "cox" | modelType == "cox_time") & length(unique(outcomes$y)) > 2) {
            stop("Cox model only accepts one outcome type")
        }
# <<<<<<< HEAD
#         if (modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron"){
#             if (!isSorted(outcomes,c("stratumId","rowId"))){
#                 if(!quiet) {
#                     writeLines("Sorting outcomes by stratumId and rowId")
#                 }
#                 rownames(outcomes) <- NULL #Needs to be null or the ordering of ffdf will fail
#                 outcomes <- outcomes[ff::ffdforder(outcomes[c("stratumId","rowId")]),]
#             }
#             if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
#                 if(!quiet) {
#                     writeLines("Sorting covariates by covariateId, stratumId and rowId")
#                 }
#                 rownames(covariates) <- NULL #Needs to be null or the ordering of ffdf will fail
#                 covariates <- covariates[ff::ffdforder(covariates[c("covariateId", "stratumId","rowId")]),]
#             }
# =======
        if ("subjectId" %in% colnames(outcomes)) {
            if (!isSorted(outcomes,
                          c("stratumId", "time", "y", "subjectId", "rowId"),
                          c(TRUE, FALSE, TRUE, TRUE, TRUE))) {
                if (!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y, subjectId and rowId")
                outcomes <- outcomes[order(outcomes$stratumId, -outcomes$time, outcomes$y, outcomes$subjectId, outcomes$rowId),]
            }
        } else {
            if (!isSorted(outcomes,
                          c("stratumId", "time", "y", "rowId"),
                          c(TRUE, FALSE, TRUE, TRUE))) {
                if (!quiet)
                    writeLines("Sorting outcomes by stratumId, time (descending), y and rowId")
                    outcomes <- outcomes[order(outcomes$stratumId, -outcomes$time, outcomes$y, outcomes$rowId),]
            }
        }
        if (!"time" %in% colnames(covariates)) {
            covariates$time <- NULL
            covariates$y <- NULL
            covariates$stratumId <- NULL
            covariates <- merge(covariates, outcomes, by = c("rowId"))
        }
        if (!isSorted(covariates,
                      c("covariateId", "stratumId", "time", "y", "rowId"),
                      c(TRUE, TRUE, FALSE, TRUE, TRUE))) {
            if (!quiet)
                writeLines("Sorting covariates by covariateId, stratumId, time (descending), y, and rowId")
            covariates <- covariates[order(covariates$covariateId, covariates$stratumId, -covariates$time, covariates$y, covariates$rowId),]
        }
    }

    if (checkRowIds) {
        mapping <- match(covariates$rowId,outcomes$rowId)
        if (any(is.na(mapping))) {
            if (!quiet)
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            covariateRowsWithMapping <- which(!is.na(mapping))
            covariates <- covariates[covariateRowsWithMapping,]
        }
    }

    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    loadNewSqlCyclopsDataY(object = dataPtr,
                           stratumId = if ("stratumId" %in% colnames(outcomes)) outcomes$stratumId else NULL,
                           rowId = outcomes$rowId,
                           y = outcomes$y,
                           time = if ("time" %in% colnames(outcomes)) outcomes$time else NULL)

    if (addIntercept & (modelType != "cox" & modelType != "cox_time" & modelType != "fgr")) {
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    }

    covarNames <- unique(covariates$covariateId)
    loadNewSqlCyclopsDataMultipleX(object = dataPtr,
                                   covariateId = covariates$covariateId,
                                   rowId = covariates$rowId,
                                   covariateValue = covariates$covariateValue,
                                   name = covarNames)

    if (modelType == "cox_time" && !is.null(timeEffectMap)) {
        if (!all(timeEffectMap$covariateId %in% covarNames)) stop("Invalid covariateId for time effects.")
        loadNewSqlCyclopsDataStratTimeEffects(object = dataPtr,
					      stratumId = outcomes$stratumId,
					      rowId = outcomes$rowId,
					      subjectId = outcomes$subjectId,
					      timeEffectCovariateId = sort(timeEffectMap$covariateId))
    }

    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if ("weights" %in% colnames(outcomes)) {
        dataPtr$weights <- outcomes$weights
    } else {
        dataPtr$censorWeights <- NULL
    }

    if ("censorWeights" %in% colnames(outcomes)) {
        dataPtr$censorWeights <- outcomes$censorWeights
    } else {
        if (modelType == "fgr") {
            dataPtr$censorWeights <- getFineGrayWeights(outcomes$time, outcomes$y)$weights
            writeLines("Generating censoring weights")
        } else {
            dataPtr$censorWeights <- NULL
        }
    }

    return(dataPtr)
}

#' @describeIn convertToCyclopsData Convert data from two \code{Andromeda} tables
#' @export
# <<<<<<< HEAD
# convertToCyclopsData.data.frame <- function(outcomes,
#                                             covariates,
#                                             modelType = "lr",
#                                             addIntercept = TRUE,
#                                             checkSorting = TRUE,
#                                             checkRowIds = TRUE,
#                                             normalize = NULL,
#                                             quiet = FALSE,
#                                             floatingPoint = 64){
#     if ((modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron") & addIntercept){
#         if(!quiet)
#             warning("Intercepts are not allowed in conditional models, removing intercept",call.=FALSE)
# =======
convertToCyclopsData.tbl_dbi <- function(outcomes,
                                         covariates,
                                         modelType = "lr",
					 timeEffectMap = NULL,
                                         addIntercept = TRUE,
                                         checkSorting = NULL,
                                         checkRowIds = TRUE,
                                         normalize = NULL,
                                         quiet = FALSE,
                                         floatingPoint = 64) {
    if (!is.null(checkSorting))
        warning("The 'checkSorting' argument has been deprecated. Sorting is now always checked")

    if ((modelType == "clr" | modelType == "cpr") & addIntercept) {
        if (!quiet) {
            warning("Intercepts are not allowed in conditional models, removing intercept", call. = FALSE)
        }
# >>>>>>> master
        addIntercept = FALSE
    }

    if (modelType == "pr" | modelType == "cpr") {
        if (any(pull(outcomes, time) <= 0)) {
            stop("time cannot be non-positive", call. = FALSE)
        }
    }

    providedNoStrata <- !"stratumId" %in% colnames(outcomes)

# <<<<<<< HEAD
#         if (modelType == "clr" | modelType == "cpr" | modelType == "clr_exact" | modelType == "clr_efron"){
#             if (!isSorted(outcomes,c("stratumId","rowId"))){
#                 if(!quiet)
#                     writeLines("Sorting outcomes by stratumId and rowId")
#                 outcomes <- outcomes[order(outcomes$stratumId,outcomes$rowId),]
#             }
#             if (!isSorted(covariates,c("covariateId", "stratumId","rowId"))){
#                 if(!quiet)
#                     writeLines("Sorting covariates by covariateId, stratumId, and rowId")
#                 covariates <- covariates[order(covariates$covariateId, covariates$stratumId,covariates$rowId),]
#             }
# =======
    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr") {
        if (providedNoStrata) {
            outcomes <- outcomes %>%
                mutate(stratumId = 0)
            covariates <- covariates %>%
                mutate(stratumId = 0)
# >>>>>>> master
        }
    }

    if (checkRowIds) {
        covariateRowIds <- covariates %>%
            distinct(.data$rowId) %>%
            pull()
        outcomeRowIds <- select(outcomes, "rowId") %>%
            pull()
        mapping <- match(covariateRowIds, outcomeRowIds)
        if (any(is.na(mapping))) {
            if (!quiet) {
                writeLines("Removing covariate values with rowIds that are not in outcomes")
            }
            covariates <- covariates %>%
                filter(.data$rowId %in% outcomeRowIds)
        }
    }

    # Sorting should be last, as other operations may change ordering.
    # Also, should always explicitly define sorting, else not guaranteed.
    if (modelType == "lr" | modelType == "pr") {
        outcomes <- outcomes %>%
            arrange(.data$rowId)

        covariates <- covariates %>%
            arrange(.data$covariateId, .data$rowId)
    }

    if (modelType == "clr" | modelType == "cpr") {
        outcomes <- outcomes %>%
            arrange(.data$stratumId, .data$rowId)

        covariates <- covariates %>%
            arrange(.data$covariateId, .data$stratumId, .data$rowId)
    }

    if (modelType == "cox" | modelType == "cox_time" | modelType == "fgr") {

        if ((modelType == "cox" | modelType == "cox_time") &
            (select(outcomes, "y") %>% distinct() %>% count() %>% collect() > 2)) {
            stop("Cox model only accepts one outcome type")
        }
        if (!"time" %in% colnames(covariates)) {
            covariates <- covariates %>%
                inner_join(select(outcomes, .data$rowId, "time", "y"), by = "rowId")
        }
        if ("subjectId" %in% colnames(outcomes)) {
            outcomes <- outcomes %>%
                arrange(.data$stratumId, desc(.data$time), .data$y, .data$subjectId, .data$rowId)
        } else {
            outcomes <- outcomes %>%
                arrange(.data$stratumId, desc(.data$time), .data$y, .data$rowId)
	}
        covariates <- covariates %>%
            arrange(.data$covariateId, .data$stratumId, desc(.data$time), .data$y, .data$rowId)
    }

    dataPtr <- createSqlCyclopsData(modelType = modelType, floatingPoint = floatingPoint)

    outcomes <- collect(outcomes)
    if (modelType == "lr" | modelType == "pr") {
        outcomes$stratumId <- NULL
    }

    loadNewSqlCyclopsDataY(object = dataPtr,
                           stratumId = if ("stratumId" %in% colnames(outcomes)) outcomes$stratumId else NULL,
                           rowId = outcomes$rowId,
                           y = outcomes$y,
                           time = if ("time" %in% colnames(outcomes)) (select(outcomes, time) %>% pull()) else NULL)

    if (addIntercept & (modelType != "cox" & modelType != "cox_time" & modelType != "fgr")) {
        loadNewSqlCyclopsDataX(dataPtr, 0, NULL, NULL, name = "(Intercept)")
    }

    loadCovariates <- function(batch) {
        covarNames <- unique(batch$covariateId)
        loadNewSqlCyclopsDataMultipleX(object = dataPtr,
                                        covariateId = batch$covariateId,
                                        rowId = batch$rowId,
                                        covariateValue = batch$covariateValue,
                                        name = covarNames,
                                        append = TRUE)
    }

    Andromeda::batchApply(covariates,
                          loadCovariates,
                          batchSize = 100000) # TODO Pick magic number

    if (modelType == "cox_time" && !is.null(timeEffectMap)) {
        covarNames <- unique(pull(covariates, .data$covariateId))
        if (!all(timeEffectMap$covariateId %in% covarNames)) stop("Invalid covariateId for time effects.")
        loadNewSqlCyclopsDataStratTimeEffects(object = dataPtr,
					      stratumId = outcomes$stratumId,
					      rowId = outcomes$rowId,
					      subjectId = outcomes$subjectId,
					      timeEffectCovariateId = sort(timeEffectMap$covariateId))
    }

    if (modelType == "pr" || modelType == "cpr")
        finalizeSqlCyclopsData(dataPtr, useOffsetCovariate = -1)

    if (!is.null(normalize)) {
        .normalizeCovariates(dataPtr, normalize)
    }

    if ("weights" %in% colnames(outcomes)) {
        dataPtr$weights <- outcomes %>% pull(.data$weights)
    } else {
        dataPtr$weights <- NULL
    }

    if ("censorWeights" %in% colnames(outcomes)) {
        dataPtr$censorWeights <- outcomes %>% pull(.data$censorWeights)
    } else {
        if (modelType == "fgr") {
            dataPtr$censorWeights <- getFineGrayWeights(
                outcomes %>% pull("time"),
                outcomes %>% pull("y")
            )$weights
            writeLines("Generating censoring weights")
        } else {
            dataPtr$censorWeights <- NULL
        }
    }

    return(dataPtr)
}
