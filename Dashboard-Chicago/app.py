#Creamos el archivo de la APP en el interprete principal (Python)

##########################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd

##########################################
#Definimos la instancia
@st.cache_resource


##########################################
#Creamos la funcion de carga de datos
def load_data():
    #Lectura del archivo csv
    df =pd.read_csv("50_sin_nulos_ni_atipicos_Chicago_Illinois_UnitedStates.csv")

    #Selecciono las columnas tipo numericas del dataframe
    numeric_df = df.select_dtypes(['float', 'int'])   #Devuelve columnas
    numeric_cols = numeric_df.columns                         #Devuelve lista de columnas

    #Selecciono las columnas tipo texto del dataframe
    text_df = df.select_dtypes(['object'])   #Devuelve columnas
    text_cols = text_df.columns                         #Devuelve lista de columnas

    #Selecciono algunas columnas categoricas de valores para desplegar en diferentes 
    categorical_column_sex = df['room_type']
    #Obtengo los valores unicos de la columna categorica seleccionada
    unique_categories_room_type = categorical_column_sex.unique()

    return df, numeric_cols, text_cols, unique_categories_room_type, numeric_df

##########################################
#Cargo los datos obtenidos de la funcion "load_data"

df, numeric_cols, text_cols, unique_categories_room_type, numeric_df = load_data()

##########################################
#CREACION DEL DASHBOARD
#Generamos las paginas que utilizaremos en el diseno
#Widget 1: Selectbox
#Menu desplegable de opciones de las paginas seleccionadas
View = st.selectbox(label = "View", options = ["Modelado explicativo", "Modelado predictivo", "View 3", "View 4"])

##########################################
#CONTENIDO DE LA VISTA 1

if View == "Modelado explicativo":
#Generamos los encabezados para el dashboard
    st.title("TITANIC")
    st.header("Panel principal")
    st.subheader("Line Plot")
##########################################
#Generamos los encabezados para la barra lateral (sidebar)
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("Sidebar")
    st.sidebar.subheader("Panel de eleccion")
##########################################
#Widget 2: Checkbox
#Generamos un cuadro de seleccion (checkbox) en una barra lateral (sidebar) para mostrar el dataset
    check_box = st.sidebar.checkbox(label = "Mostrar Dataset")

#Condicional para que aparezca el dataframe
    if check_box:
        #Mostramos el dataset
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())
##########################################
#Widget 3: Multiselect box
#Generamos un cuadro de multi-seleccion (Y) para seleccionar variables a graficar
    numerics_vars_selected = st.sidebar.multiselect(label = "Variables graficadas", options = numeric_cols)

##########################################
#Widget 3: Selectbox
#Menu desplegable de opciones de la variable categorica seleccionada
    category_selected = st.sidebar.selectbox(label = "Categorias", options = unique_categories_room_type)

##########################################
#Widget 4: Button
#Generamos un button (Button) en la barra lateral (sidebar) para mostrar las variables tipo texto
    Button = st.sidebar.button(label = "Mostrar variables STRING")

#Condicional para que aparezca el dataframe
    if Button:
#Mostramos el dataset
        st.write(text_cols)

##########################################
#Graph 1: LINEPLOT
#Despliegue de un line plot, definiendo las variables "X categorias y Y numericas"
    data = df[df['room_type'] == category_selected]
    data_features = data[numerics_vars_selected]
    figure1 = px.line(data_frame = data_features, x = data_features.index, 
                            y = numerics_vars_selected, title = str('Features of rooms'), 
                            width = 1600, height =600)

#Generamos un button (Button) en la barra lateral (sidebar) para mostrar el lineplot
    Button2  = st.sidebar.button(label = "Mostrar grafica tipo lineplot")

#Condicional para que aparezca la grafica tipo line plot
    if Button2:
#Mostramos el lineplot
        st.plotly_chart(figure1)

##########################################
#Contenido de la vista 2
elif View == "Modelado predictivo":
#Generamos los encabezados para el dashboard
    st.title("Titatic")
    st.header("Panel principal")
    st.subheader("Scatter plot")

##########################################
#Graph 2: Scatterplot
    x_selected = st.sidebar.selectbox(label = "x", options = numeric_cols)
    y_selected = st.sidebar.selectbox(label = "y", options = numeric_cols)
    figure2 = px.scatter(data_frame = numeric_df, x = x_selected, y = y_selected, title = 'Dispersiones')
    st.plotly_chart(figure2)

##########################################
#Contenido de la vista 3
elif View == "View 3":
#Generamos los encabezados para el dashboard
    st.title("Titatic")
    st.header("Panel principal")
    st.subheader("Pie plot")

#Menu desplegable de opciones de las variables seleccionadas
    Variable_cat = st.sidebar.selectbox(label = "Variable categorica", options = text_cols)
    Variable_num = st.sidebar.selectbox(label = "Variable numerica", options = numeric_cols)

##########################################
#Graph 3: Pieplot
#Despliegue de un pieplot, definiendo las variables "X categoricas" y "Y numericas"
    figure3 = px.pie(data_frame = df, names = df[Variable_cat], 
                            values = df[Variable_num], title = str('Features of ') + ' ' + 'rooms', 
                            width = 1600, height = 600)
    st.plotly_chart(figure3)

##########################################
#Contenido de la vista 4
elif View == "View 4":
#Generamos los encabezados para el dashboard
    st.title("Titatic")
    st.header("Panel principal")
    st.subheader("Bar plot")

#Menu desplegable de opciones de las variables seleccionadas
    Variable_cat = st.sidebar.selectbox(label = "Variable categorica", options = text_cols)
    Variable_num = st.sidebar.selectbox(label = "Variable numerica", options = numeric_cols)

##########################################
#Graph 4: Barplot
#Despliegue de un barplot, definiendo las variable "X categoricas" y "Y numericas"
    figure4 = px.bar(data_frame = df, x = df[Variable_cat], y = df[Variable_num], 
                            title =  str('Features of ') + ' ' + 'rooms')
    figure4.update_xaxes(automargin = True)
    figure4.update_yaxes(automargin = True)
    st.plotly_chart(figure4)
