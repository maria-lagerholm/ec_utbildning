Blocket Car Analysis
================
Geisol Yissel Urbina and Maria Lagerholm

- [Introduction](#introduction)

# Introduction

In this project, we explore the Swedish used car market using data
scraped from Blocket and enriched with technical attributes from
Transportstyrelsen and external statistics from SCB. The analysis has
two core objectives:

To predict market prices for mid-range used cars in Sweden using machine
learning (e.g., Random Forest).

To understand what variables actually influence insurance premiums and
vehicle tax levels via statistical inference (e.g., linear regression).

We combine predictive and explanatory modeling to better understand
price-setting mechanisms and cost structures for private car ownership.
The figure below, generated from SCB data, illustrates trends in new car
registrations, which reflect macro-level market shifts that may
indirectly affect the availability and pricing of used cars.

------------------------------------------------------------------------

``` r
library(remotes)
#remotes::install_github("rOpenGov/pxweb", force = TRUE, build_vignettes = FALSE)
library(pxweb)
```

    ## pxweb 0.17.1: R Interface to PXWEB APIs.
    ## https://github.com/ropengov/pxweb

``` r
Sys.setlocale(locale = "UTF-8")
```

    ## [1] "en_US.UTF-8/UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8"

``` r
# Step 1: Read your JSON file into a proper query object
px_query <- pxweb_query("scb/trafik.json")

# Step 2: Download the data using the parsed query object
px_data <- pxweb_get(
  url = "https://api.scb.se/OV0104/v1/doris/sv/ssd/TK/TK1001/TK1001S/SnabbStatTK1001",
  query = px_query
)

# Step 3: Convert to data frame
px_data_frame <- as.data.frame(px_data, column.name.type = "text", variable.value.type = "text")
```

``` r
library(ggplot2)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
# Om du redan har laddat ner datan till t.ex. `px_data_frame`
# Annars kan du skapa ett exempeldata manuellt så här:
df <- px_data_frame

# Byt gärna till enklare kolumnnamn
colnames(df) <- c("manad", "forandring")

# Om du inte redan gjort detta:
df$datum <- as.Date(paste0(substr(df$manad, 1, 4), "-", substr(df$manad, 6, 7), "-01"))

ggplot(df, aes(x = datum, y = forandring)) +
  geom_point(size = 1.5) +
  geom_smooth(method = "loess", span = 0.2, color = "darkred", se = FALSE, size = 1) +
  theme_minimal() +
  labs(
    title = "Trend för förändring i nyregistrerade bilar",
    x = "Datum",
    y = "Förändring jämfört med året innan (%)"
  )
```

    ## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
    ## ℹ Please use `linewidth` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

    ## `geom_smooth()` using formula = 'y ~ x'

![](scb_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->
