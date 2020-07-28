library(dplyr)
library(tidyr)
library(data.table)
library(echarts4r)
library(echarts4r.maps)
library(glue)
library(googleVis)

get.data <- function(url, var.name){
  df <- read.csv(url, stringsAsFactors = F)%>%
    reshape2::melt(id.vars = c("Province.State","Country.Region",
                               "Lat","Long" ), 
                   variable.name = 'date', 
                   value.name = 'confirmed') %>%
    mutate(date = as.Date(date, 'X%m.%d.%y')) %>%
    filter(!is.na(confirmed)) %>%
    mutate(Country.Region = as.character(Country.Region),
           Country.Region = case_when(
             Country.Region == 'US' ~ 'United States',
             Country.Region == 'Taiwan*' ~ 'Taiwan',
             Country.Region == 'Korea, South' ~ 'South Korea',
             TRUE ~ Country.Region)) %>%
    group_by(Province.State, Country.Region, Lat, Long, date) %>%
    summarise(confirmed = max(confirmed)) %>%
    group_by(Country.Region, Province.State, date, Lat, Long) %>%
    summarise(confirmed = sum(confirmed))
  colnames(df)[6] <- var.name
  return(df)
}

plot.country.trend <- function(df.country, country = 'Singapore', new = F){
  if (new){
    df.country %>%
      filter(Country.Region == country) %>%
      e_charts(date) %>% # initialise and set x
      e_line(confirmed_new, smooth = TRUE, name = 'confirmed') %>%  # add a line
      e_area(recovered_new, smooth = TRUE, name = 'recovered') %>%  # add area
      e_area(death_new, smooth = TRUE, name = 'death') %>%  # add area
      #e_title("Country trend", glue('{country} - new')) %>%  # Add title & subtitle
      e_theme("infographic") %>%  # theme
      e_legend(right = 0) %>%  # move legend to the bottom
      e_tooltip(trigger = "axis") # tooltip  
  } else {
    df.country %>%
      filter(Country.Region == country) %>%
      e_charts(date) %>% # initialise and set x
      e_line(confirmed, smooth = TRUE, name = 'confirmed') %>%  # add a line
      e_area(recovered, smooth = TRUE, name = 'recovered') %>%  # add area
      e_area(death, smooth = TRUE, name = 'death') %>%  # add area
      #e_title("Country trend", glue('{country} - total')) %>%  # Add title & subtitle
      e_theme("infographic") %>%  # theme
      e_legend(right = 0) %>%  # move legend to the bottom
      e_tooltip(trigger = "axis") # tooltip
  }
  
}

plot.country.map <- function(df, country = 'China', show.anim = F, pieces, province.map.cn){
  
  if(country == 'China'){
    country.list <- c('China', 'Taiwan','Hong Kong','Taipei and environs')
  } else {
    country.list <- c(country)
  }
  
  if(country == 'China'){
    res <- df %>%
      ungroup() %>%
      filter(Country.Region %in% country.list) %>%
      filter(confirmed > 0) %>%
      mutate(Province.State = as.character(Province.State)) %>%
      left_join(province.map.cn)
  } else {
    res <- df %>%
      ungroup() %>%
      filter(Country.Region %in% country.list) %>%
      filter(confirmed > 0) %>%
      mutate(name = as.character(Province.State))  
  }
  
  if(show.anim){
    res %>%
      mutate(month = format(as.Date(date), '%Y-%m')) %>%
      group_by(month) %>%      
      e_charts(name, timeline = T) %>%
      e_timeline_opts(autoPlay = T, playInterval = 1000) %>%
      em_map(country) %>%
      e_map(confirmed, map = country, name = 'confirmed case') %>%
      e_visual_map(confirmed, 
                   type = 'piecewise',
                   pieces = pieces) %>%
      e_theme('infographic') %>%
      e_title(glue('Propagation in {country}')) %>%
      e_tooltip(formatter = e_tooltip_choro_formatter(
        style = 'decimal'))
  } else {
    res %>%
      filter(date == max(res$date)) %>%
      e_charts(name) %>%
      em_map(country) %>%
      e_map(confirmed, map = country, name = 'confirmed case') %>%
      e_visual_map(confirmed, 
                   type = 'piecewise',
                   pieces = pieces) %>%
      e_theme('infographic') %>%
      e_title(glue('Propagation in {country}')) %>%
      e_tooltip(formatter = e_tooltip_choro_formatter(
        style = 'decimal'))
  }
  
}

plot.world.map <- function(df.country, pieces, show.anim = F){
  
  df.country <- df.country %>%
    ungroup() %>%
    mutate(Country.Region = case_when(
      Country.Region == 'South Korea' ~ 'Korea',
      TRUE ~ Country.Region
    ))
  
  if(show.anim){
    df.country %>%
      ungroup() %>%
      mutate(Country.Region = as.character(Country.Region)) %>%
      filter(confirmed > 0) %>%
      mutate(month = format(as.Date(date), '%Y-%m')) %>%
      group_by(month) %>%  
      e_charts(Country.Region, timeline = T) %>%
      e_timeline_opts(autoPlay = T, playInterval = 1000) %>%
      e_map(confirmed, map = 'world', name = 'confirmed case') %>%
      e_visual_map(confirmed, 
                   type = 'piecewise',
                   pieces = pieces) %>%
      e_tooltip(formatter = e_tooltip_choro_formatter()) %>%
      e_title('Propagation Worldwide') %>%
      e_theme('infographic')
  } else {
    df.country %>%
      ungroup() %>%
      mutate(Country.Region = as.character(Country.Region)) %>%
      filter(confirmed > 0 & date == max(df.country$date)) %>%
      e_charts(Country.Region) %>%
      e_map(confirmed, map = 'world', name = 'confirmed case') %>%
      e_visual_map(confirmed, 
                   type = 'piecewise',
                   pieces = pieces) %>%
      e_tooltip(formatter = e_tooltip_choro_formatter()) %>%
      e_title('Propagation Worldwide') %>%
      e_theme('infographic')
  }
  
}

get.kpi <- function(df.country, country = NULL){
  
  if(is.null(country)){
    res <- df.country %>%
      filter(date == max(df.country$date))
  } else {
    res <- df.country %>%
      filter(Country.Region == country) %>%
      filter(date == max(df.country$date))
  }
  
  updated.date <- max(df.country$date)
  
  confirmed <- sum(res$confirmed, na.rm = T)
  recovered <- sum(res$recovered, na.rm = T)
  death <- sum(res$death, na.rm = T)
  
  confirmed_new <- sum(res$confirmed_new, na.rm = T)
  recovered_new <- sum(res$recovered_new, na.rm = T)
  death_new <- sum(res$death_new, na.rm = T)
  
  return(list(
    confirmed = confirmed,
    recovered = recovered,
    death = death,
    confirmed_new = confirmed_new,
    recovered_new = recovered_new,
    death_new = death_new,
    updated.date = updated.date,
    mortality.rate = death / confirmed
  ))
  
}

fill.na <- function(i, x) ifelse(is.na(i), x, i)

plot.world.table <- function(df.country, 
                             pageSize = 20, width = 1200,
                             newly = F){
  
  res <- df.country %>% 
    ungroup() %>%
    filter(date == max(df.country$date)) %>%
    arrange(desc(confirmed)) %>%
    left_join(Population %>% select(Country, Flag), 
              by = c('Country.Region' = 'Country')) %>%
    select(-date) %>%
    mutate(Country.Region = paste(Flag,  Country.Region)) %>%
    select(-Flag,) %>%
    mutate(Mortality.Rate = death / confirmed)
  
  if(newly){
    # new number updated today
    res <- res %>%
      select(`country/region` = Country.Region, 
             confirmed = confirmed_new, 
             recovered = recovered_new, 
             death = death_new)
    gvisTable(res,
              formats=list(confirmed="#,###",
                           death = "#,###"),
              options = list(page = 'enable',
                             pageSize = pageSize,
                             width = width))
    
  } else {
    res <- res %>%
      select(`country/region` = Country.Region, 
             confirmed, 
             recovered, 
             death, 
             `morality rate` = Mortality.Rate)
    gvisTable(res,
              formats=list(confirmed="#,###",
                           death = "#,###",
                           `morality rate` = "#.#%"),
              options = list(page = 'enable',
                             pageSize = pageSize,
                             width = width))
  }
}
