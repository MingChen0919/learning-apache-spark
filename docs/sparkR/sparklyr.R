# connecting to spark
library(sparklyr)
sc = spark_connect(master = "local")

# using dplyr
install.packages(c('nycflights13', 'Lahman'))

library(dplyr)
iris_tbl = copy_to(sc, iris)
flights_tbl = copy_to(sc, nycflights13::flights, 'flights')
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")
src_tbls(sc)

delay <- flights_tbl %>% 
  group_by(tailnum) %>%
  summarise(count = n(), dist = mean(distance), delay = mean(arr_delay)) %>%
  filter(count > 20, dist < 2000, !is.na(delay)) %>%
  collect


library(DBI)
iris_preview = dbGetQuery(sc, "SELECT * FROM iris LIMIT 10")
iris_preview


spark_disconnect(sc)
