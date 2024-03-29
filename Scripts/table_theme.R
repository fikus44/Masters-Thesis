
library("knitr")
library("kableExtra")


table_theme = function(data, colnames, caption, escape = TRUE) {
  
  data %>%
    kable(format = "latex", #Output er kompatibelt med LaTeX
          digits = 4,  # no. of digits after comma
          col.names = colnames, #s�jlenavne
          align = c("l", rep("c", length(colnames) - 2), "r"), #S�jle alignement, f�rste s�jle er venstre aligned, herefter er alle s�jle p� n�r den sidste s�jle center aligned. Den sidste s�jle er h�jre aligned
          caption = caption, #Tabel titel. Husk at caption skal angives som string i funktion. 
          format.args = list(big.mark = ",", scientific = FALSE), #Tilf�jer  1.000 tals separator. Scientific = FALSE betyder vi IKKE benytter os af e til at denote tal
          escape = escape, #Output tager h�jde for special characters i LaTeX som f.eks. bliver _ til \_
          booktabs = TRUE, #Benytter os af booktabs package i LaTeX
          linesep =  '' #Intet ekstra line space hver x. linje. Kan �ndres ved c("", "", "","","\\addlinespace")
            
          )
}


#Til input:

# Data skal v�re dataframe eller tibble
# colnames skal v�re vektor, hvor l�ngden af vektor er lig med antallet af s�jler i data
# caption skal v�re angivet som string

# Dokumentation:

# https://bookdown.org/yihui/rmarkdown-cookbook/kable.html
# https://haozhu233.github.io/kableExtra/awesome_table_in_pdf.pdf

# Tables kan customizes yderligere ved brug af bl.a. kable_styling() funktionen.
# Jeg vil foresl�, man ser p� dokuemntation, hvis der er et specifikt customization
# problem som tabellen skal im�dekomme. 

#Tester table_theme

data(mtcars)
test = head(mtcars)
names = c("mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb")

table_theme(test, colnames = names, caption = "test af table_theme funktionen") %>%
  kable_styling(latex_options = "scale_down") #Scaler table til margins i LaTeX
