#'show_quantile
#'
#' calculate the quantiles for all numeric columns
#'
#' @param df dataframe that quantile function performed on
#' @param q quantiles, defaults to 0 to 1, with step of 0.1
#' @return list
#' @export
#' @examples
#' data(iris)
#' show_quantile(iris)
show_quantile <- function(df, q = seq(0,1,0.1)){
  res <- list()
  for(v in colnames(df)){
    if(is.numeric(df[,v])){
      res[[v]] <- quantile(df[,v],q)
    }
  }
  return(res)
}

#' show_n_levels
#'
#' calculate number of levels in each categorical field
#'
#' @param df input dataframe
#' @return list
#' @importFrom dplyr n_distinct
#' @export
#' @examples
#' data(iris)
#' show_n_levels(iris)
show_n_levels <- function(df){
  res <- list()
  for(v in colnames(df)){
    if(is.character(df[,v])){
      res[[v]] <- n_distinct(df[,v])
    }
    if(is.factor(df[,v])){
      res[[v]] <- length(levels(df[,v]))
    }
  }
  return(res)
}
