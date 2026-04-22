import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Segmentador K-Means", layout="wide")
st.title("Segmentador Dinámico Universal")
st.markdown("Sube cualquier dataset. Aplica filtros, limpia los datos automáticamente y agrupa seleccionando tus variables.")

# --- Memoria de la App ---
if 'modelo_entrenado' not in st.session_state:
    st.session_state['modelo_entrenado'] = False

# --- Barra Lateral ---
st.sidebar.header("Parámetros")
k_elegido = st.sidebar.slider("Clusters (K):", 2, 10, 3)

# --- Carga de Archivos ---
archivo_subido = st.file_uploader("Selecciona tu archivo (.csv o .xlsx)", type=["csv", "xlsx"])

if archivo_subido is not None:
    @st.cache_data
    def cargar_datos(archivo):
        try:
            if archivo.name.endswith('.csv'):
                return pd.read_csv(archivo, encoding='latin1')
            return pd.read_excel(archivo)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return None

    df_crudo = cargar_datos(archivo_subido)
    st.success("Archivo cargado correctamente.")
    st.dataframe(df_crudo.head(3))

    st.markdown("---")
    
    # ==========================================
    # SELECCIÓN ESTRICTA DE 2 COLUMNAS
    # ==========================================
    st.subheader("Selección de Variables (Ejes)")
    st.write("El modelo requiere **sí o sí** dos variables numéricas para trazar la gráfica.")
    
    columnas = df_crudo.columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        col_id = st.selectbox("ID o Etiqueta (Opcional)", options=["Ninguno"] + columnas)
    with col2:
        col_x = st.selectbox("Variable Numérica 1 (Eje X)", options=columnas, index=0)
    with col3:
        col_y = st.selectbox("Variable Numérica 2 (Eje Y)", options=columnas, index=min(1, len(columnas)-1))

    if col_x == col_y:
        st.warning("Por favor, elige dos variables diferentes para X e Y.")
        st.stop()

    # ==========================================
    # EL TOQUE MAESTRO: FILTRO OPCIONAL
    # ==========================================
    st.markdown("---")
    st.subheader("Pre-filtro de Datos (Opcional)")
    usar_filtro = st.checkbox("¿Deseas analizar solo un segmento específico? (Ej. Un solo país, o un solo producto)")
    
    if usar_filtro:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            col_filtro = st.selectbox("Selecciona la columna para filtrar:", options=columnas)
        with f_col2:
            valores_unicos = df_crudo[col_filtro].dropna().unique()
            valor_filtro = st.selectbox("Selecciona el valor específico a analizar:", options=valores_unicos)

    st.markdown("---")

    # --- Ejecución del Modelo ---
    if st.button("Aplicar Limpieza y Ejecutar K-Means"):
        st.session_state['modelo_entrenado'] = True

    if st.session_state['modelo_entrenado']:
        with st.spinner("Procesando datos y calculando centroides..."):
            
            df_limpio = df_crudo.copy()
            
            # 1. Aplicamos el filtro si el usuario lo activó
            if usar_filtro:
                df_limpio = df_limpio[df_limpio[col_filtro] == valor_filtro]
            
            # 2. Forzamos la conversión a números
            df_limpio[col_x] = pd.to_numeric(df_limpio[col_x], errors='coerce')
            df_limpio[col_y] = pd.to_numeric(df_limpio[col_y], errors='coerce')
            
            # 3. Limpiamos nulos
            df_limpio = df_limpio.dropna(subset=[col_x, col_y])
            
            hover_name = col_id if col_id != "Ninguno" else None

            if len(df_limpio) < k_elegido:
                st.error(f"Error: Después de filtrar y limpiar, solo quedan {len(df_limpio)} registros válidos. Necesitas al menos {k_elegido} (tu valor de K) para agrupar.")
                st.stop()

            # Extraemos las variables
            tensor = df_limpio[[col_x, col_y]]

            # Escalamiento y Modelo
            scaler = StandardScaler()
            datos_escalados = scaler.fit_transform(tensor)

            modelo = KMeans(n_clusters=k_elegido, random_state=42, n_init=10)
            df_limpio['Cluster'] = modelo.fit_predict(datos_escalados)
            df_limpio['Cluster_Label'] = df_limpio['Cluster'].astype(str)

            st.success(f"¡Proceso exitoso! Se analizaron {len(df_limpio)} registros válidos.")
            
            # Gráficos Dinámicos
            titulo_grafico = f'Segmentación (K={k_elegido})'
            if usar_filtro:
                titulo_grafico += f' - Filtrado por: {valor_filtro}'

            fig = px.scatter(
                df_limpio, x=col_x, y=col_y, color='Cluster_Label',
                hover_name=hover_name,
                title=titulo_grafico,
                opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            centroides = scaler.inverse_transform(modelo.cluster_centers_)
            fig.add_trace(go.Scatter(
                x=centroides[:, 0], y=centroides[:, 1], mode='markers', 
                marker=dict(color='black', symbol='x', size=15, line=dict(width=3)), name='Centroides'
            ))
            st.plotly_chart(fig, width='stretch')

            # Métricas
            st.subheader("Análisis de Volatilidad por Segmento")
            cluster_seleccionado = st.selectbox("Selecciona un segmento:", options=sorted(df_limpio['Cluster'].unique()))
            
            datos_x = df_limpio[df_limpio['Cluster'] == cluster_seleccionado][col_x]
            datos_y = df_limpio[df_limpio['Cluster'] == cluster_seleccionado][col_y]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Varianza ({col_x})", f"{datos_x.var():,.2f}")
            c2.metric(f"Desv. Est. ({col_x})", f"{datos_x.std():,.2f}")
            c3.metric(f"Varianza ({col_y})", f"{datos_y.var():,.2f}")
            c4.metric(f"Desv. Est. ({col_y})", f"{datos_y.std():,.2f}")

            # Reporte
            st.markdown("---")
            st.subheader("Reporte de Convergencia")
            st.info(
                f"**¿Cómo se movieron los centroides?**\n\n"
                f"Al configurar $K={k_elegido}$, el algoritmo posicionó inicialmente {k_elegido} "
                f"centroides de manera aleatoria. Durante el entrenamiento, calculó la distancia Euclidiana "
                f"de cada registro hacia estos puntos, asignándolos al grupo más cercano. \n\n"
                f"Tras cada asignación, los centroides se desplazaron geométricamente hacia la media exacta "
                f"de su nuevo grupo. Este ciclo iterativo redujo progresivamente la inercia del sistema hasta "
                f"alcanzar la **Convergencia**: el punto de equilibrio final."
            )

else:
    st.info("Por favor, sube tu archivo para comenzar.")