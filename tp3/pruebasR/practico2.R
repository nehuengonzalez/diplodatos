normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x)))
}
#Tomamos datos de D\'olar futuro tomado de ROFEX. Tomamos como separador el tab \t
data <- read.csv('activos.xls', sep='\t')  

#Transformamos las fechas String en formato Dates
data$Fecha <- as.Date(data$Fecha, format="%d/%m/%Y")

#Reemplazamos "," por "." y luego convertimos en numero las columnas Ultimo, Primero, Minimo y Maximo.
data$Primero <- as.numeric(gsub(",", ".", data$Primero))
data$Ultimo <- as.numeric(gsub(",", ".", data$Ultimo))
data$Minimo <- as.numeric(gsub(",", ".", data$Minimo))
data$Maximo <- as.numeric(gsub(",", ".", data$Maximo))

#Filtramos solo 3 grupos de activos: DLR052019, RFX20062019 y ORO052019
data$Posicion <- gsub(" ", "", data$Posicion)
data$Posicion
dolarFiltro <- subset(data, (gsub(" ", "", data$Posicion) == "DLR052019") | (gsub(" ", "", data$Posicion) == "RFX20062019") | (gsub(" ", "", data$Posicion) == "ORO052019"))

plot(dolarFiltro$Volumen, dolarFiltro$Fecha, main="Posiciones Activos")
dolarFiltro$Posicion
dolarFiltro

#Sacamos ahora las posiciones del set de datos a procesar y las almacenamos.
posiciones <- dolarFiltro$Posicion
dataSinPosicion <- dolarFiltro[-2]

#Normalizamos los valores de la columna Ultimo
library(Hmisc)
dolarFiltro2 <- dolarFiltro[5:8]
dolarFiltro2
dolarFiltro2 <- impute(dolarFiltro2)
dolarFiltroNormalizado <- as.data.frame(lapply(dolarFiltro2, scale))
#dolarFiltroNormalizado <- dolarFiltro2
plot(dolarFiltroNormalizado$Volumen, dolarFiltroNormalizado$Volumen, main="Todas las posiciones")
dolarFiltro2
dolarFiltroNormalizado

#Tomamos 1000 valores aleatorios del dataset para usar como train data
limiteInf = length(dolarFiltroNormalizado[,1])/4
limiteSup = length(dolarFiltroNormalizado[,1])

data_train <- dolarFiltroNormalizado[1:limiteInf,]
data_test <- dolarFiltroNormalizado[(limiteInf + 1):limiteSup,]

data_train_labels <- dolarFiltro$Posicion[1:limiteInf]
data_test_labels <- dolarFiltro$Posicion[(limiteInf+1):limiteSup]
data_train_labels

library(class)
data_test_pred <- knn(train=data_train, test=data_test, cl=data_train_labels, k=13)

library(gmodels)
CrossTable(x=data_test_labels, y=data_test_pred, prop.chisq = FALSE)

#Filtramos solo 3 grupos de activos: DLR052019, DLR082019 y DLR122019
data$Posicion <- gsub(" ", "", data$Posicion)
data$Posicion
dolarFiltro <- subset(data, (gsub(" ", "", data$Posicion) == "DLR052019") | (gsub(" ", "", data$Posicion) == "DLR082019") | (gsub(" ", "", data$Posicion) == "DLR122019"))

plot(dolarFiltro$Volumen, dolarFiltro$Fecha, main="Posiciones Activos")
dolarFiltro$Posicion
dolarFiltro

#Sacamos ahora las posiciones del set de datos a procesar y las almacenamos.
posiciones <- dolarFiltro$Posicion
dataSinPosicion <- dolarFiltro[-2]

#Normalizamos los valores de la columna Ultimo
library(Hmisc)
dolarFiltro2 <- dolarFiltro[5:8]
dolarFiltro2 <- impute(dolarFiltro2)
dolarFiltroNormalizado <- as.data.frame(lapply(dolarFiltro2, scale))
plot(dolarFiltroNormalizado$Primero, dolarFiltroNormalizado$Ultimo, main="Todas las posiciones")
dolarFiltro2
dolarFiltroNormalizado

#Tomamos 1000 valores aleatorios del dataset para usar como train data
limiteInf = length(dolarFiltroNormalizado[,1])/5
limiteSup = length(dolarFiltroNormalizado[,1])

data_train <- dolarFiltroNormalizado[1:limiteInf,]
data_test <- dolarFiltroNormalizado[(limiteInf + 1):limiteSup,]

data_train_labels <- dolarFiltro$Posicion[1:limiteInf]
data_test_labels <- dolarFiltro$Posicion[(limiteInf+1):limiteSup]
data_train_labels

library(class)
data_test_pred <- knn(train=data_train, test=data_test, cl=data_train_labels, k=35)

library(gmodels)
CrossTable(x=data_test_labels, y=data_test_pred, prop.chisq = FALSE)

