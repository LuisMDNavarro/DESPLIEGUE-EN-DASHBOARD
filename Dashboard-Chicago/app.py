#Creamos el archivo de la APP en el interprete principal (Python)

##########################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_option_menu import option_menu
from funpymodeling.exploratory import freq_tbl
import numpy as np

##########################################
#Definimos la instancia
@st.cache_resource


##########################################
#Creamos la funcion de carga de datos
def load_data():
    #Lectura del archivo csv
    df =pd.read_csv("50_sin_nulos_ni_atipicos_Chicago_Illinois_UnitedStates.csv")
    df = df.drop(['Unnamed: 0'], axis = 1)

    #Selecciono las columnas tipo numericas del dataframe
    numeric_df = df.select_dtypes(['float', 'int'])   #Devuelve columnas
    numeric_cols = numeric_df.columns                         #Devuelve lista de columnas

    #Selecciono las columnas tipo texto del dataframe
    text_df = df.select_dtypes(['object'])   #Devuelve columnas
    text_cols = text_df.columns                         #Devuelve lista de columnas

    #Selecciono algunas columnas categoricas de valores para desplegar en sus diferentes categorias
    categorical_column_room_type = df['room_type']
    #Obtengo los valores unicos de la columna categorica seleccionada
    unique_categories_room_type = categorical_column_room_type.unique()

    return df, numeric_cols, text_cols, unique_categories_room_type, numeric_df, text_df

##########################################
#Cargo los datos obtenidos de la funcion "load_data"

df, numeric_cols, text_cols, unique_categories_room_type, numeric_df, text_df = load_data()

##########################################
#Dashboard

st.set_page_config(layout="wide")
#Navbar
View = option_menu(
    menu_title=None,  # Oculta el título
    options= ["Inicio", "Modelado explicativo", "Modelado predictivo"],
    icons=["house", "graph-up", "cpu"],  # Íconos de Bootstrap
    menu_icon="cast",
    default_index=1,
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

    #Variable para tipo de variable a graficar
    if 'variable_type' not in st.session_state:
        st.session_state.variable_type = 'numeric'

    #Titulos y encabezados
    st.title("Airbnb Chicago")
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("⚙️ Opciones")

    #Cambiar entre numericas y categoricas
    st.sidebar.subheader("🧬 Variable")
    button_type_variable = st.sidebar.button(label = "Cambiar tipo de variable")
    if button_type_variable:
        if st.session_state.variable_type == 'numeric':
            st.session_state.variable_type = 'categoric'
        else:
            st.session_state.variable_type = 'numeric'
    
    if st.session_state.variable_type == 'numeric':
        category_variable_selected = st.sidebar.selectbox(label = "Variables categoricas", options = text_cols)
        table = freq_tbl(df[category_variable_selected])
    else:
        numeric_variable_selected = st.sidebar.selectbox(label = "Variables numericas", options = numeric_cols)
        #Categorizar la variable numerica
        dataNumeric = df.copy()
        n = 8269
        Max = dataNumeric[numeric_variable_selected].max()
        Min = dataNumeric[numeric_variable_selected].min()
        R = Max - Min
        ni = max(5, min(12, round(1 + 3.32 * np.log10(n))))
        intervalos = np.linspace(Min, Max, ni + 1)
        categorias = [f"Intervalo de: {intervalos[i]:.2f} a {intervalos[i+1]:.2f}" for i in range(len(intervalos) - 1)]
        dataNumeric[numeric_variable_selected] = pd.cut(x = dataNumeric[numeric_variable_selected], bins = intervalos, labels = categorias)
        table = freq_tbl(dataNumeric[numeric_variable_selected])

    #Cambiar la frecuencia para los graficos
    st.sidebar.subheader("🔍 Filtro de frecuencia")
    frequency = st.sidebar.number_input("Frecuencia: " , min_value=0, max_value=None, value=0, step=None, format="%d")

    #Mostrar analisis univariado
    st.sidebar.subheader("🧪 Análisis univariado")
    check_box_analysis = st.sidebar.checkbox(label = "Mostrar analisis")
    if check_box_analysis:
        #Obtengo un analisis univariado de una variable en especifico
        if st.session_state.variable_type == 'numeric':
            st.header("Análisis univariado de: " + category_variable_selected)
        else:
            st.header("Análisis univariado de: " + numeric_variable_selected)
        table2 = table[table['frequency'] > frequency]
        st.write(table2)

    #Mostrar graficos
    st.sidebar.subheader("Graficos 📊")
    check_box_line = st.sidebar.checkbox(label = "📈 Grafico de lineas")
    check_box_bars = st.sidebar.checkbox(label = "📊 Grafico de barras")
    check_box_scatter = st.sidebar.checkbox(label = "🟢 Grafico de dispersion")
    check_box_area = st.sidebar.checkbox(label = "📉 Grafico de area")
    check_box_pie = st.sidebar.checkbox(label = "🥧 Grafico de pastel")


    if  check_box_line or check_box_bars or check_box_scatter or check_box_area or check_box_pie:
        if st.session_state.variable_type == 'numeric':
            st.header("Graficos de: " + category_variable_selected)
        else: 
            st.header("Graficos de: " + numeric_variable_selected)

    if check_box_line:
        st.subheader("Line Plot")
        if st.session_state.variable_type == 'numeric':
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.line(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.line(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)

        st.plotly_chart(figure1)

    if check_box_bars:
        st.subheader("Bar Plot")
        if st.session_state.variable_type == 'numeric':
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.bar(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", color= category_variable_selected, width = 1600, height =600)
        else:
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.bar(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", color= numeric_variable_selected, width = 1600, height =600)

        st.plotly_chart(figure1)

    if check_box_scatter:
        st.subheader("Scatter Plot")
        if st.session_state.variable_type == 'numeric':
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.scatter(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.scatter(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)

        st.plotly_chart(figure1)

    if check_box_area:
        st.subheader("Area Plot")
        if st.session_state.variable_type == 'numeric':
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.area(data_frame = Filtro, x = category_variable_selected, 
            y = "frequency", width = 1600, height =600)
        else:
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.area(data_frame = Filtro, x = numeric_variable_selected, 
            y = "frequency", width = 1600, height =600)

        st.plotly_chart(figure1)

    if check_box_pie:
        st.subheader("Pie Plot")
        if st.session_state.variable_type == 'numeric':
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.pie(data_frame = Filtro, names = category_variable_selected, 
            values = "frequency", width = 1600, height =600)
        else:
            table2 = table.drop(['percentage', 'cumulative_perc'], axis = 1)
            Filtro = table2[table2['frequency'] > frequency]
            figure1 = px.pie(data_frame = Filtro, names = numeric_variable_selected, 
            values = "frequency", width = 1600, height =600)

        st.plotly_chart(figure1)

    
    #Mostrar Dataset
    st.sidebar.subheader("ℹ️ Dataset")
    check_box = st.sidebar.checkbox(label = "Mostrar Dataset")
    if check_box:
        st.subheader("Dataset info:")
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())

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
