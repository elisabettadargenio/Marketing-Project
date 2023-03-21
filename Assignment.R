rm(list=ls())

#se pacchetti non sono già installati usare il comando install.packages("nome_pacchetto")

#loading dei pacchetti
library(caret)
library(corrplot)
library(DataExplorer)
library(rpart)
library(rpart.plot)
library(nnet)
library(ipred)
library(psych)
library(gam)
library(randomForest)

# Set della working directory in cui sono contenuti i dati e i filenames

working_dir = "C:/Users/dargenioe/Documents/Private"
filename1 = "merchandinsingProductData.csv"
filename2 = "webMarketingProductData.csv"
filename3 = "testSet.csv"

setwd(working_dir)

#Leggiamo dati Merchandising
merch_data<-read.csv(filename1)
head(merch_data)
length(unique(merch_data$Product))
#una riga per ogni product

#Leggiamo dati Marketing
marketing_data<-read.csv(filename2)
head(marketing_data)

#uniamo i due dataset tramite id del Prodotto
merged<-merge(merch_data,marketing_data, by.x="Product", by.y = "Product")

head(merged)
str(merged)
summary(merged)
#tutte variabili continue già scalate, tranne l'id del prodotto

#Check NA
table(is.na(merged)) #no NA



#rimuovo variabile ID del Product, non è modellabile
data<-subset(merged, select = -Product)
head(data)


################# MD vs CR ########################

#analisi della relazione tra MD e CR

pairs.panels(data[,c("MD","CR")], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)
#La correlazione indica la tendenza che hanno due variabili a variare insieme.
#Una correlazione positiva indica che all'aumentare dell'una aumenta anche l'altra.
#più l'indice di correlazione si avvicina ad 1, più la relazione è forte ed è approssimabile
#da una retta, più si avvicina a 0 più la relazione è debole. qui 0.8, molto correlata
#Anche dalla distribuzione congiunta sembra esserci relazione lineare, infatti il modello di regressione lineare
#fitta quasi perfettamente i dati

#adattiamo un modello lineare per spiegare la relazione tra MD e CR
mod1<-lm(data$CR~data$MD)
summary(mod1)
#per ogni aumento unitario di MD ho un aumento di quasi 4 di CR 
#il modello spiega il 63% della variabilità totale


##### Analisi esplorativa #####


#Correlazione tra tutte le variabili

corrplot(cor(data), method="ellipse",type="upper") 
#molte covariate sembrano correlate tra loro
#in particolare PPC e MD; Price e AIV ; PPC e Budget

#inoltre emerge una forte correlazione tra la variabile risposta e molte delle covariate, 
#soprattutto AIV (correlazione negativa), MD e PPC (positiva)

#vediamo con più precisione le correlazioni
plot_correlation(data)
#non ci sono variabili collineari
#PPC e MD sono fortemente correlate con la risposta

#esaminiamo meglio la relazione tra MD e PPC perchè si sospetta che MD potrebbe avere impatto su PPC
pairs.panels(data[,c("MD","PPC")], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)
#risulta una correlazione di 0.59, c'è una relazione quasi lineare tra le due variabili
#tendenzialmente all'aumentare della percentuale di sconto aumentano 
#le spese di web marketing sui canali di performance


##### Stima-Verifica ######
set.seed(1234)

#separiamo i dati in train e validation, il train serve per allenare il modello, il validation per avere
#delle misure sulla bontà di adattamento del modello ai dati

#scegliamo di usare il 75% delle osservazioni per il train e il restante per il validation
samp<- sample(1:nrow(data), nrow(data)/100*75)

train<-data[samp,]
validation<-data[-samp,]

#vediamo andamento delle variabili con la risposta (escludiamo MD che è stata studiata in precedenza)
par(mfrow=c(2,3))
for (i in c(2,4:8)){
  plot(train[,i],train$CR, ylab = "CR", xlab = colnames(train)[i])
}

#sembra non esserci una relazione troppo forte tra la propensity to buy 
#e le variabili Price, Display, Budget e Competitor

par(mfrow=c(1,1))

##### Modello Lineare #####

#il primo modello adattato è un modello lineare in cui vengono incluse tutte le covariate a disposizione
m0 <- lm(CR~., data=train)  
summary(m0)
#sembra essere un ottimo modello, abbiamo un R^2 di 0.96, quindi quasi tutta la variabilità della risposta è spiegata
#dalle variabili esplicative

#in particolare abbiamo 4 variabili molto significative: AIV, MD, PPC e Display

#previsioni con il modello sui dati di validation
p0 <- predict(m0, newdata=validation)

#calcoliamo Mean Squared Error e Mean Absolute Error (per tutti i modelli)
mse0<- mean((validation$CR - p0)^2)
mae0<-mean(abs(validation$CR-p0))


##### GAM (Generalized Additive Models) #####

#I GAM sono di fatto un'estensione non-parametrica del modello linere.
#adattiamo delle spline di lisciamento alle variabili quantitative

m1 <- gam(CR~s(AIV)+s(MD)+s(Price)+s(PPC)+s(Display)+s(Budget)+ s(Competitor), data = train)
summary(m1)
#nessun effetto non parametrico risulta significativo, tra gli effetti parametrici si confermano significative
#AIV, MD, PPC e Display

p1 <- predict(m1, newdata = validation, type="response")
mse1<-mean((validation$CR-p1)^2)
mae1<-mean(abs(validation$CR-p1))


##### Albero di Regressione #####

#Internamente, rpart tiene traccia della complessità dell'albero.
#La misura della complessità è una combinazione delle dimensioni di un albero e 
#della capacità di previsione dell'albero. 
#Se la successiva migliore suddivisione nella crescita di un albero non riduce la complessità complessiva 
#dell'albero di una certa quantità, rpart terminerà il processo di crescita.

tree <- rpart(CR ~ ., data = train)
rpart.plot(tree, extra = "auto")
summary(tree) # MD è la variabile più importante 
printcp(tree) #

p2 <- predict(tree, validation)


mse2<- mean((p2-validation$CR)^2)
mae2<-mean(abs(p2-validation$CR))


##### Rete Neurale #####

#procedimento di stima e verifica su alcune combinazioni di numero di nodi latenti e $\lambda$ (parametro 
#di penalizzazione della funzione obiettivo),
#selezionando il numero di nodi latenti e il valore del parametro di penalizzazione 
#che minimizzano la devianza penalizzata

parte1 <- sample(1:nrow(train), nrow(train)/2)
parte2 <- setdiff(1:nrow(train), parte1)

decay=10^seq(-3,-1,length=3)
nodi=c(3,5)
err=matrix(0,length(decay),length(nodi))
colnames(err) <- nodi
rownames(err) <- decay
for(i in 1:length(decay)){
  for(j in 1:length(nodi))  {
    model_rn=nnet(CR~.,data=train[parte1,],size=nodi[j],decay=decay[i],linout=T,maxNWts=2000,maxit=800)
    pred_rn=pmax(predict(model_rn,newdata=train[parte2,]),min(train[parte2,]$CR))
    err[i,j]=mean((train[parte2,]$CR-pred_rn)^2)
    print(paste("Nodo",nodi[j]))
  }
  print(paste("Decay",decay[i]))
}
err
coord_rn <- which(err==min(err), arr.ind=TRUE)
nodi[coord_rn[2]] ; decay[coord_rn[1]]
model_rn=nnet(CR~.,data=train,size=nodi[coord_rn[2]],decay=decay[coord_rn[1]],linout=T,maxNWts=2000,maxit=1000) #lo stesso
pred_rn=predict(model_rn,newdata=validation)
mrn<-mean((validation$CR-pred_rn)^2)
mrna<-mean(abs(validation$CR-pred_rn))


##### Bagging #######
#Il bagging è una versione boostrap dei CART.
#un campione casuale di dati in un set di addestramento viene selezionato con reinserimento

set.seed(123)

bag<-bagging(train$CR~. , data=train[,-1], nbagg=50, coob=T) 

pBag <- predict(bag, newdata = validation) 
mseBag <- mean((pBag-validation$CR)^2)
maeBag<- mean(abs(pBag-validation$CR))



####### Random Forest #######

#combinazioni di alberi, in ciascuno split, oltre ad una selezione casuale delle osservazioni, 
#lo split è determinato utilizzando un sottoinsieme di variabili, scelte casualmente.

m6 <- randomForest(x=train[,-1], y=train$CR , ntree=50,
                   do.trace = T)
plot(m6)
importance(m6)
varImpPlot(m6)
#variabili più importanti sono PPC, MD e AIV

p6<- predict(m6, newdata = validation)
eRf<- mean((p6-validation$CR)^2)
eRFa<-mean(abs(p6-validation$CR))



##### Confronto Modelli #####
nomi<- c("Modello lineare", "GAM", 
         "Albero di regressione", 
         "Rete Neurale", "Bagging", "Random Forest")
confronto=cbind(nomi, round(c(mse0,mse1,
                              mse2,
                              mrn,mseBag, eRf),2),
                round(c(mae0,mae1,
                        mae2,
                        mrna,maeBag, eRFa),2))
colnames(confronto)=c("modello","MSE","MAE")
confronto

#l'albero di regressione e il bagging performano peggio degli altri
#il modello lineare performa come il MARS e la rete neurale, e leggermente meglio della random forest.
#Viene selezionato come modello finale il modello lineare per semplicità e interpretabilità dei risultati
#useremo quindi il modello lineare per fare previsione sui nuovi dati.

##### Previsioni sui dati nuovi #######

test_data <- read.csv(filename3)
test_data$CR <- predict(m0, test_data)

head(test_data)

write.csv(test_data, file="new_testSet.csv", row.names = FALSE)
