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
activos <- subset(data, (gsub(" ", "", data$Posicion) == "DLR052019") | (gsub(" ", "", data$Posicion) == "RFX20062019") | (gsub(" ", "", data$Posicion) == "ORO052019"))

plot(activos$Volumen, activos$Fecha, main="Posiciones Activos")
activos$Posicion
activos

#Sacamos ahora las posiciones del set de datos a procesar y las almacenamos.
posiciones <- activos$Posicion
dataSinPosicion <- activos[-2]

#Normalizamos los valores de la columna Ultimo
library(Hmisc)
OHLC <- activos[5:8]
OHLC
OHLC <- impute(OHLC)
dataNorm <- as.data.frame(lapply(OHLC, scale))
#dolarFiltroNormalizado <- dolarFiltro2
plot(dataNorm$Ultimo, dataNorm$Fecha, main="Todas las posiciones")
activos
dataNorm

#Tomamos 1000 valores aleatorios del dataset para usar como train data
limiteInf = round(length(dataNorm[,1])/4)
limiteSup = length(dataNorm[,1])

data_train <- dataNorm[1:limiteInf,]
data_test <- dataNorm[(limiteInf + 1):limiteSup,]

data_train_labels <- activos$Posicion[1:limiteInf]
data_test_labels <- activos$Posicion[(limiteInf+1):limiteSup]
data_train_labels

library(class)
data_test_pred <- knn(train=data_train, test=data_test, cl=data_train_labels, k=13)

library(gmodels)
CrossTable(x=data_test_labels, y=data_test_pred, prop.chisq = FALSE)

#Filtramos solo 3 grupos de activos: DLR052019, DLR082019 y DLR122019
data$Posicion <- gsub(" ", "", data$Posicion)
data$Posicion
dolares <- subset(data, (gsub(" ", "", data$Posicion) == "DLR052019") | (gsub(" ", "", data$Posicion) == "DLR082019") | (gsub(" ", "", data$Posicion) == "DLR122019"))

plot(dolares$Volumen, dolares$Fecha, main="Posiciones Activos")
dolares$Posicion
dolares

#Sacamos ahora las posiciones del set de datos a procesar y las almacenamos.
posiciones <- dolares$Posicion
dataSinPosicion <- dolares[-2]

#Normalizamos los valores de la columna Ultimo
library(Hmisc)
OHLC <- dolares[5:8]
OHLC <- impute(OHLC)
dataNor <- as.data.frame(lapply(OHLC, scale))
plot(dataNor$Primero, dataNor$Ultimo, main="Todas las posiciones")
OHLC$Ultimo
OHLC

#Tomamos 1000 valores aleatorios del dataset para usar como train data
limiteInf = length(dataNor[,1])/5
limiteSup = length(dataNor[,1])

data_train <- dataNor[1:limiteInf,]
data_test <- dataNor[(limiteInf + 1):limiteSup,]

data_train_labels <- dolares$Posicion[1:limiteInf]
data_test_labels <- dolares$Posicion[(limiteInf+1):limiteSup]
data_train_labels

library(class)
data_test_pred <- knn(train=data_train, test=data_test, cl=data_train_labels, k=5)

library(gmodels)
CrossTable(x=data_test_labels, y=data_test_pred, prop.chisq = FALSE)


