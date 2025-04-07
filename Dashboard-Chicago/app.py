#Creamos el archivo de la APP en el interprete principal (Python)

##########################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_option_menu import option_menu

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
#Dashboard

st.set_page_config(layout="wide")
#Navbar
View = option_menu(
    menu_title=None,  # Oculta el título
    options= ["Inicio", "Modelado explicativo", "Modelado predictivo"],
    icons=["house", "graph-up", "cpu"],  # Íconos de Bootstrap
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

##########################################
#Index

if View == "Inicio":
    st.title("Airbnb, Chicago  Illinois")
    st.write("Este dashboard presenta un Modelado explicativo usando un análisis univariado "
                        "de las variables categóricas más significativas y un Modelado predictivo usando "
                        "un análisis aplicando regresión lineal simple, regresión lineal multiple y regresión logistica,"
                        "esto haciendo uso de los datos propios de Airbnb acerca de la ciudad de Chicago Illinois, EU.")
    img, title = st.columns([1, 7])
    left, right = st.columns([3, 1])
    img.image("img/airbnb.png", width=80)
    title.header("Acerca de Airbnb")
    left.subheader("¿Que es?")
    left.write("Airbnb es una plataforma digital que conecta a personas que desean alquilar su propiedad "
                            "(total o parcialmente) con viajeros que buscan alojamiento temporal. Fundada en 2008, "
                            "Airbnb ha transformado la industria del hospedaje, ofreciendo alternativas más flexibles y "
                            "personalizadas que los hoteles tradicionales.A través de su modelo de economía colaborativa, "
                            "permite que anfitriones publiquen espacios disponibles y que huéspedes puedan reservarlos de "
                            "forma segura, utilizando filtros como precio, ubicación, tipo de propiedad, calificaciones, y más.")
    right.image("img/airbnb.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                                    - Opera en más de 220 países y regiones.
                                    - Más de 4 millones de anfitriones en todo el mundo.
                                    - Más de 150 millones de usuarios han reservado a través de la plataforma.
                                    - Ofrece desde alojamientos económicos hasta opciones de lujo (Airbnb Luxe).
                                    """)
    img, title = st.columns([1, 7])
    left, right = st.columns([1, 3])
    img.image("img/usa.png", width=80)
    title.header("Acerca de Chicago")
    right.subheader("¿Por qué Chicago?")
    right.write("Chicago, ubicada en el estado de Illinois, es la tercera ciudad más grande de Estados Unidos y uno "
                        "de los destinos turísticos y culturales más importantes del país. Con una arquitectura emblemática, "
                        "una escena artística vibrante y una rica historia, la ciudad atrae a millones de visitantes cada año.")
    right.write("En el contexto de Airbnb, Chicago representa un mercado urbano dinámico con una gran diversidad "
                                "de alojamientos, desde apartamentos modernos en el centro hasta casas históricas en barrios "
                                "residenciales. Su perfil turístico, junto con eventos internacionales y zonas de alta demanda como "
                                "The Loop, Lincoln Park o Wicker Park, la convierten en un punto clave para el análisis de comportamiento "
                                "en plataformas de hospedaje.")
    left.image("img/chicago.jpg", width=300)
    st.subheader("Datos relevantes:")
    st.markdown("""
                                    - 📍 Ubicación: Estado de Illinois, Estados Unidos
                                    - 🌆 Población: Aproximadamente 2.7 millones de habitantes.
                                    - 🗺️ Ubicación estratégica: A orillas del lago Míchigan, con vistas panorámicas y actividades acuáticas.
                                    - ✈️ Fácil acceso internacional: El Aeropuerto O’Hare es uno de los más transitados del mundo, con vuelos a casi todos los continentes.
                                    - 🏙️ Principales atracciones:
                                        - Millennium Park (con el famoso "Bean")
                                        - Willis Tower (Skydeck con piso de vidrio)
                                        - Art Institute of Chicago (uno de los mejores museos del mundo)
                                        - Riverwalk (paseo a lo largo del río Chicago)
                                        - Navy Pier (zona de entretenimiento junto al lago)
                                    - 🍕 Gastronomía icónica:
                                        - Pizza estilo Chicago (deep-dish)
                                        - Hot dogs "Chicago-style"
                                        - Gran oferta multicultural en barrios como Pilsen, Chinatown y Little Italy
                                    """)

##########################################
#Modelado explicativo

elif View == "Modelado explicativo":
#Generamos los encabezados para el dashboard
    st.title("Air bnb Chicago")
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
    st.title("Air bnb Chicago")
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
    st.title("Air bnb Chicago")
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
    st.title("Air bnb Chicago")
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
