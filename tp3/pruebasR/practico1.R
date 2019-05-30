#1)

library(nycflights13)
flights <- nycflights13::flights
flights

retardados<-subset(flights, flights$arr_delay > 120)
retardados
hist(retardados$arr_delay, freq = F, main = "Histograma de Vuelos Retardados", xlim = c(0, 600), las = 1)

#2)
toHouston<-subset(flights, (flights$dest == "IAH") | (flights$dest == "HOU"))
toHouston
pie(table(toHouston$dest), freq = F, main = "Pie Vuelos a Houston")

#3)
unitedDeltaAmerican<-subset(flights, (flights$carrier == "AA") | (flights$carrier == "DL") | (flights$carrier == "UA"))
unitedDeltaAmerican$carrier 
pie(table(unitedDeltaAmerican$carrier), main = "Pie Carriers")

#4)
departureJulioAgostoSeptiembre<-subset(flights, (flights$month == 7) | (flights$month == 8) | (flights$month == 9))
prueba<-subset(departureJulioAgostoSeptiembre, departureJulioAgostoSeptiembre$month == 9)
pie(table(departureJulioAgostoSeptiembre$month))
hist(table(departureJulioAgostoSeptiembre$month), freq = F, main = "", breaks = 2, las=1)
table(departureJulioAgostoSeptiembre$month)
prueba

#5)
retardados2<-subset(flights, (flights$arr_delay > 120) & (flights$dep_delay <= 0))
retardados2
hist(departureJulioAgostoSeptiembre$arr_delay, freq = F, main= "Histograma Demorados pero salieron a tiempo")

#6)
entreCeroYseis<-subset(flights, flights$dep_time <= 600)
entreCeroYseis
subset(entreCeroYseis, entreCeroYseis$dep_time < 100)
hist((entreCeroYseis$dep_time), freq = F, main = "")

