import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FormatStrFormatter

# Configuración inicial de estilo
plt.style.use(hep.style.CMS)

def merge_event_tables_with_ratio(tables: dict) -> pd.DataFrame:
    """Une tablas de eventos y calcula ratios."""
    merged = None
    x_col, y_col = None, None

    for name, df in tables.items():
        df_copy = df.copy()
        cols = df_copy.columns.tolist()

        # Detectar columnas de binning
        bin_cols = [col for col in cols if col.lower() != "events"]
        if len(bin_cols) != 2:
            raise ValueError(f"❌ La tabla '{name}' no tiene exactamente 2 columnas de binning.")

        x_tmp, y_tmp = bin_cols
        if x_col is None:
            x_col, y_col = x_tmp, y_tmp

        # Renombrar columna 'events'
        df_copy = df_copy.rename(columns={"events": name})

        # Unir tablas
        if merged is None:
            merged = df_copy
        else:
            merged = pd.merge(merged, df_copy, on=[x_col, y_col], how="outer")

    merged = merged.fillna(0)

    # Mover fila Total al final
    is_total = (merged[x_col] == "Total") & (merged[y_col] == "Total")
    total_row = merged[is_total]
    merged = merged[~is_total]

    # Calcular ratio
    if all(k in merged.columns for k in ["Total", "MainBgr", "Data"]):
        numerator = merged["Data"] - (merged["Total"] - merged["MainBgr"])
        denominator = merged["MainBgr"]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denominator != 0, numerator / denominator, np.nan)
        merged["2D_weights"] = ratio
    else:
        print("❌ No se encontraron todas las columnas necesarias: Total, MainBgr, Data.")

    # Ordenar y reinsertar Total
    merged = merged.sort_values(by=[x_col, y_col]).reset_index(drop=True)
    if not total_row.empty:
        merged = pd.concat([merged, total_row], ignore_index=True)

    return merged

def adjust_table_with_ratio_range(
    df: pd.DataFrame,
    ratio_col: str = "2D_weights",
    ratio_range: tuple = (0.7, 1.4),
    min_data_events: int = 5
) -> pd.DataFrame:
    """Ajusta bins dinámicamente basado en ratios."""
    df = df.copy()
    bin_cols = [col for col in df.columns if df[col].dtype == "object"]
    
    if len(bin_cols) != 2:
        raise ValueError(f"❌ No se pudieron detectar 2 columnas de binning.")
    
    x_bin_col, y_bin_col = bin_cols
    df = df[(df[x_bin_col] != "Total") & (df[y_bin_col] != "Total")].reset_index(drop=True)
    df = sort_by_bin_start(df, y_bin_col)

    adjusted_rows = []
    grouped = df.groupby(x_bin_col, sort=False)

    for x_val, group in grouped:
        group = group.reset_index(drop=True)
        i = 0
        n = len(group)
        
        while i < n:
            current_group = [group.iloc[i]]
            total = group.iloc[i]["Total"]
            mainbgr = group.iloc[i]["MainBgr"]
            data = group.iloc[i]["Data"]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (data - (total - mainbgr)) / mainbgr if mainbgr != 0 else np.nan
            
            j = i + 1
            while j < n:
                condition_ratio = (np.isnan(ratio) or (ratio_range[0] <= ratio <= ratio_range[1]))
                condition_data = (data >= min_data_events)
                
                if condition_ratio and condition_data:
                    break
                
                current_group.append(group.iloc[j])
                total += group.iloc[j]["Total"]
                mainbgr += group.iloc[j]["MainBgr"]
                data += group.iloc[j]["Data"]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = (data - (total - mainbgr)) / mainbgr if mainbgr != 0 else np.nan
                
                j += 1
            
            # Procesar bins combinados
            y_bins = [row[y_bin_col] for row in current_group]
            lefts, rights = zip(*[(
                float(b.split(',')[0][1:]),
                float(b.split(',')[1][:-1])
            ) for b in y_bins])
            
            combined_row = {
                x_bin_col: x_val,
                y_bin_col: f"[{min(lefts)}, {max(rights)})",
                "Total": total,
                "MainBgr": mainbgr,
                "Data": data,
                ratio_col: ratio if (data >= min_data_events) else 1.0
            }
            adjusted_rows.append(combined_row)
            i = j

    adjusted_df = pd.DataFrame(adjusted_rows)
    
    # Añadir fila Total
    total_row = {
        x_bin_col: "Total",
        y_bin_col: "Total",
        "Total": adjusted_df["Total"].sum(),
        "MainBgr": adjusted_df["MainBgr"].sum(),
        "Data": adjusted_df["Data"].sum(),
        ratio_col: np.nan
    }

    return pd.concat([adjusted_df, pd.DataFrame([total_row])], ignore_index=True)

def sort_by_bin_start(df: pd.DataFrame, bin_col: str = None) -> pd.DataFrame:
    """Ordena por valor inicial del bin."""
    if bin_col is None:
        bin_col = df.columns[0]

    def extract_start(bin_str):
        if bin_str == "Total":
            return float('inf')
        return float(bin_str.strip("[]()").split(",")[0])
    
    return (df.copy()
            .assign(_bin_start=df[bin_col].map(extract_start))
            .sort_values(by="_bin_start")
            .drop(columns="_bin_start")
            .reset_index(drop=True))

def sort_table_by_bin_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por bordes de bins X e Y."""
    df = df.copy()
    bin_cols = [col for col in df.columns if df[col].dtype == "object"]
    
    if len(bin_cols) != 2:
        raise ValueError(f"No se encontraron 2 columnas tipo bin: {bin_cols}")
    
    x_col, y_col = bin_cols
    is_total = (df[x_col] == "Total") & (df[y_col] == "Total")
    df_total = df[is_total]
    df = df[~is_total]

    def extract_start(bin_str):
        if bin_str == "Total":
            return float('inf')
        return float(bin_str.strip("[]()").split(",")[0])

    df_sorted = (df.copy()
                 .assign(_x_start=df[x_col].map(extract_start),
                            _y_start=df[y_col].map(extract_start))
                 .sort_values(by=["_x_start", "_y_start"])
                 .drop(columns=["_x_start", "_y_start"])
                 .reset_index(drop=True))

    if not df_total.empty:
        df_sorted = pd.concat([df_sorted, df_total], ignore_index=True)

    return df_sorted

def get_lumi_text(year) -> str:
    """Genera etiqueta de luminosidad para el año."""
    lumi_map = {
        "2017": "41.5",
        "2018": "59.8", 
        "2016": "16.8",
        "2016APV": "19.5"
    }
    lumi = lumi_map.get(str(year), "XX.X")
    return f"{lumi} fb$^{{-1}}$ (13 TeV)"

def plot_weight_heatmap_from_table(
    df,
    year="2018",
    value_col="2D_weights",
    output="",
    figsize=(12, 10),
    fontsize=14,
    title="2D Weights"
):
    import matplotlib.pyplot as plt
    import numpy as np
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    plt.close('all')

    NODATA_COLOR = 0.8
    CMAP = plt.cm.viridis
    NODATA_ALPHA = 0.6

    df = df.copy()
    bin_cols = [col for col in df.columns if col not in ["Total", "MainBgr", "Data", value_col]]
    if len(bin_cols) != 2:
        raise ValueError("❌ Error en columnas de binning")

    x_col, y_col = bin_cols
    df = df[(df[x_col] != "Total") & (df[y_col] != "Total")]
    df = sort_by_bin_start(sort_by_bin_start(df, x_col), y_col)

    def get_edges(bin_series):
        edges = set()
        for bin_str in bin_series:
            start, end = map(float, bin_str.strip("[]()").split(","))
            edges.update([start, end])
        return np.sort(list(edges))

    x_edges = get_edges(df[x_col])
    y_edges = get_edges(df[y_col])

    matrix = np.full((len(y_edges)-1, len(x_edges)-1), np.nan)
    for _, row in df.iterrows():
        x_start, x_end = map(float, row[x_col].strip("[]()").split(","))
        y_start, y_end = map(float, row[y_col].strip("[]()").split(","))
        val = row[value_col]

        for xi in range(len(x_edges)-1):
            if x_edges[xi] >= x_start and x_edges[xi+1] <= x_end:
                for yi in range(len(y_edges)-1):
                    if y_edges[yi] >= y_start and y_edges[yi+1] <= y_end:
                        matrix[yi, xi] = val

    fig, ax = plt.subplots(figsize=figsize)

    data_values = matrix[~np.isnan(matrix)]
    vmin = np.min(data_values) if len(data_values) > 0 else 0.9
    vmax = np.max(data_values) if len(data_values) > 0 else 1.1
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 0.1, vmax + 0.1

    im = ax.pcolormesh(
        x_edges, y_edges,
        np.where(np.isnan(matrix), NODATA_COLOR, matrix),
        shading='auto', cmap=CMAP, vmin=vmin, vmax=vmax
    )

    if np.any(np.isnan(matrix)):
        ax.pcolormesh(
            x_edges, y_edges,
            np.ma.masked_array(np.ones_like(matrix), mask=~np.isnan(matrix)),
            shading='auto', cmap='Greys', alpha=NODATA_ALPHA, vmin=0, vmax=1
        )

    # Etiqueta CMS (solo "CMS Preliminary")
    hep.cms.text("Preliminary", ax=ax, fontsize=fontsize - 1)
    hep.cms.lumitext(get_lumi_text(year), ax=ax, fontsize=fontsize - 2)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x_center = (x_edges[j] + x_edges[j+1]) / 2
            y_center = (y_edges[i] + y_edges[i+1]) / 2
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'black' if val > 1.03 or val < 0.95 else 'white'
                ax.text(
                    x_center, y_center, f"{val:.2f}",
                    ha='center', va='center',
                    color=text_color, fontsize=fontsize-2,
                    fontweight='bold'
                )


    cbar = fig.colorbar(im, ax=ax, pad=0.02, format=FormatStrFormatter('%.2f'))
    cbar.set_label('2D Weight', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 3)

    ax.set_xlabel(x_col, fontsize=fontsize)
    ax.set_ylabel(y_col, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+2, pad=20)

    # Etiquetas exactas en ejes X e Y según edges
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.tick_params(axis='both', labelsize=fontsize)

    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.15)

    if output:
        path = f"{output}/weight_2D_{year}.pdf"
        plt.savefig(path, format="pdf", bbox_inches='tight', dpi=300)
        print(f"✅ Gráfico guardado en: {path}")

    plt.show()




def smooth_weights_table(
    df,
    value_col="2D_weights",
    sigma=1.0,
    output=None,
    year="2018",
    filter_range=(0.5, 2.0)
):
    """Aplica suavizado Gaussiano a los pesos."""
    df = clean_extreme_weights(df, value_col, *filter_range)
    
    bin_cols = [col for col in df.columns if col not in ["Total", "MainBgr", "Data", value_col]]
    if len(bin_cols) != 2:
        raise ValueError("❌ No se detectaron columnas de binning")
    
    x_col, y_col = bin_cols
    df = df[(df[x_col] != "Total") & (df[y_col] != "Total")]

    # Extraer bordes
    x_edges = sorted(set(
        float(b.split(',')[0][1:]) for b in df[x_col]
    ) | set(
        float(b.split(',')[1][:-1]) for b in df[x_col]
    ))
    
    y_edges = sorted(set(
        float(b.split(',')[0][1:]) for b in df[y_col]
    ) | set(
        float(b.split(',')[1][:-1]) for b in df[y_col]
    ))

    # Construir matriz
    matrix = np.ones((len(y_edges)-1, len(x_edges)-1))
    for _, row in df.iterrows():
        x_start = float(row[x_col].split(',')[0][1:])
        y_start = float(row[y_col].split(',')[0][1:])
        i = np.where(y_edges == y_start)[0][0]
        j = np.where(x_edges == x_start)[0][0]
        if not np.isnan(row[value_col]):
            matrix[i, j] = row[value_col]

    # Suavizado
    smoothed_matrix = gaussian_filter(matrix, sigma=sigma, mode='nearest')

    # Reconstruir DataFrame
    smoothed_rows = []
    for i in range(smoothed_matrix.shape[0]):
        for j in range(smoothed_matrix.shape[1]):
            smoothed_rows.append({
                x_col: f"[{x_edges[j]}, {x_edges[j+1]})",
                y_col: f"[{y_edges[i]}, {y_edges[i+1]})",
                value_col: smoothed_matrix[i, j]
            })

    df_smoothed = pd.DataFrame(smoothed_rows)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(x_edges, y_edges, smoothed_matrix, shading='auto', cmap='viridis')
    
    # Añadir estilo CMS
    hep.cms.label(
        ax=ax,
        data=False,
        label="Preliminary",
        year=year,
        lumi=get_lumi_text(year).split(" ")[0],
        fontsize=12
    )
    
    # Añadir valores
    for i in range(smoothed_matrix.shape[0]):
        for j in range(smoothed_matrix.shape[1]):
            val = smoothed_matrix[i, j]
            ax.text(
                (x_edges[j] + x_edges[j+1]) / 2,
                (y_edges[i] + y_edges[i+1]) / 2,
                f"{val:.2f}", ha='center', va='center', 
                color='black', fontsize=10
            )

    plt.colorbar(c, ax=ax, label='Smoothed 2D Weight')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Smoothed 2D Weights ({year})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output:
        filename = f"{output}/weight_2D_smoothed_{year}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches='tight')
        print(f"✅ Gráfico suavizado guardado en: {filename}")

    plt.show()
    
    return df_smoothed

def clean_extreme_weights(df, col="2D_weights", low=0.5, high=2.0):
    """Limpia valores extremos en los pesos."""
    df = df.copy()
    mask = (df[col] < low) | (df[col] > high)
    df.loc[mask, col] = 1.0
    return df

def export_weights_txt(
    df,
    X_variable,
    X_binning,
    Y_variable,
    Y_binning,
    filename="weights_2D.txt",
    value_col="2D_weights"
):
    """Exporta pesos a archivo TXT."""
    df = df[(df[X_variable] != "Total") & (df[Y_variable] != "Total")]
    bin_map = {}

    for _, row in df.iterrows():
        x_start, x_end = map(float, row[X_variable].strip("[]()").split(","))
        y_start, y_end = map(float, row[Y_variable].strip("[]()").split(","))
        val = round(float(row[value_col]), 3)

        for xi in range(len(X_binning)-1):
            if X_binning[xi] >= x_start and X_binning[xi+1] <= x_end:
                for yi in range(len(Y_binning)-1):
                    if Y_binning[yi] >= y_start and Y_binning[yi+1] <= y_end:
                        bin_map[(X_binning[xi], Y_binning[yi])] = f"{val:.3f}"

    # Escribir archivo
    with open(filename, "w") as f:
        # Cabecera
        y_labels = [f"{Y_binning[i]}-{Y_binning[i+1]}" for i in range(len(Y_binning)-1)]
        f.write(" " * 30 + " / ".join(y_labels) + "\n")

        # Filas
        for xi in range(len(X_binning)-1):
            row = []
            for yi in range(len(Y_binning)-1):
                row.append(bin_map.get((X_binning[xi], Y_binning[yi]), "1.000"))
            
            x_label = f"{X_binning[xi]}-{X_binning[xi+1]}"
            f.write(f"{x_label:<28} / " + " / ".join(row) + "\n")

    print(f"✅ Archivo TXT exportado: {filename}")

def weights_2D(
    df_total_bgr,
    df_main_bgr,
    df_data,
    ratio_range=(0.7, 1.4),
    year="2018",
    output_folder="",
    filter_params=(0.5, 2.0, 1.0),
    apply_smoothing=True,
    X_variable=None,
    X_binning=None,
    Y_variable=None,
    Y_binning=None
):
    """Función principal para cálculo de pesos 2D."""
    # Paso 1: Combinar tablas
    merged = merge_event_tables_with_ratio({
        "Total": df_total_bgr,
        "MainBgr": df_main_bgr,
        "Data": df_data
    })
    
    # Paso 2: Ajustar bins
    adjusted = adjust_table_with_ratio_range(
        merged,
        ratio_range=ratio_range
    )
    
    # Paso 3: Ordenar y visualizar
    adjusted_sorted = sort_table_by_bin_edges(adjusted)
    plot_weight_heatmap_from_table(adjusted_sorted, year=year, output=output_folder)
    
    # Guardar resultados
    if apply_smoothing:
        smoothed = smooth_weights_table(
            adjusted_sorted,
            sigma=filter_params[2],
            year=year,
            output=output_folder,
            filter_range=filter_params[:2]
        )
        smoothed.to_csv(
            f"{output_folder}/weights_2D_{year}_smoothed.csv",
            index=False, sep=";"
        )
        
        if X_variable and Y_variable and X_binning and Y_binning:
            export_weights_txt(
                smoothed,
                X_variable, X_binning,
                Y_variable, Y_binning,
                f"{output_folder}/weights_2D_{year}.txt"
            )
        return smoothed
    else:
        adjusted_sorted.to_csv(
            f"{output_folder}/weights_2D_{year}.csv",
            index=False, sep=";"
        )
        
        if X_variable and Y_variable and X_binning and Y_binning:
            export_weights_txt(
                adjusted_sorted,
                X_variable, X_binning,
                Y_variable, Y_binning,
                f"{output_folder}/weights_2D_{year}.txt"
            )
        return adjusted_sorted

    