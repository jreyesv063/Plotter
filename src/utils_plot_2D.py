import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_2d_hist_single_sample(
    sample_dict: dict,
    feature_x: str,
    feature_y: str,
    bins_x: np.ndarray,
    bins_y: np.ndarray,
    weights_key: str = "weights",
    norm_factor: float = 1.0,
    include_overflow: bool = True,
    include_underflow: bool = True,
    verbose: bool = False
):
    # Recuperar variables
    x = sample_dict.get(feature_x)
    y = sample_dict.get(feature_y)
    weights = sample_dict.get(weights_key, None)

    # Si faltan variables, retornar histograma vacío
    if x is None or y is None:
        if verbose:
            print(f"⚠️ Variables '{feature_x}' o '{feature_y}' no están disponibles. Se omite esta muestra.")
        return np.zeros((
            len(bins_x)-1 + (1 if include_overflow else 0) + (1 if include_underflow else 0),
            len(bins_y)-1 + (1 if include_overflow else 0) + (1 if include_underflow else 0)
        ))

    if weights is not None:
        weights = weights * norm_factor

    # Inicializar histograma
    nx = len(bins_x) - 1
    ny = len(bins_y) - 1
    hist2d = np.zeros((nx + include_overflow + include_underflow, 
                       ny + include_overflow + include_underflow))

    # Máscaras para underflow/overflow
    in_x = (x >= bins_x[0]) & (x < bins_x[-1])
    in_y = (y >= bins_y[0]) & (y < bins_y[-1])
    in_both = in_x & in_y

    # Histograma principal (sin underflow/overflow)
    H, _, _ = np.histogram2d(
        x[in_both], y[in_both],
        bins=[bins_x, bins_y],
        weights=weights[in_both] if weights is not None else None
    )
    
    # Posiciones de los bins centrales (sin underflow/overflow)
    central_slice = (
        slice(include_underflow, nx + include_underflow),
        slice(include_underflow, ny + include_underflow)
    )
    hist2d[central_slice] = H

    # Manejo de underflow/overflow
    if include_underflow or include_overflow:
        # Bins para underflow/overflow en X
        x_uf = x < bins_x[0]
        x_of = x >= bins_x[-1]
        x_in = ~x_uf & ~x_of

        # Bins para underflow/overflow en Y
        y_uf = y < bins_y[0]
        y_of = y >= bins_y[-1]
        y_in = ~y_uf & ~y_of

        # Underflow en X (primer bin en X)
        if include_underflow:
            # Underflow X con Y en rango
            mask = x_uf & y_in
            if weights is not None:
                hist2d[0, central_slice[1]] += np.histogram(y[mask], bins=bins_y, weights=weights[mask])[0]
            else:
                hist2d[0, central_slice[1]] += np.histogram(y[mask], bins=bins_y)[0]

            # Underflow X con underflow Y (esquina [0,0])
            mask = x_uf & y_uf
            hist2d[0, 0] += np.sum(weights[mask]) if weights is not None else np.sum(mask)

            # Underflow X con overflow Y (esquina [0,-1])
            mask = x_uf & y_of
            hist2d[0, -1] += np.sum(weights[mask]) if weights is not None else np.sum(mask)

        # Overflow en X (último bin en X)
        if include_overflow:
            # Overflow X con Y en rango
            mask = x_of & y_in
            if weights is not None:
                hist2d[-1, central_slice[1]] += np.histogram(y[mask], bins=bins_y, weights=weights[mask])[0]
            else:
                hist2d[-1, central_slice[1]] += np.histogram(y[mask], bins=bins_y)[0]

            # Overflow X with underflow Y (esquina [-1,0])
            mask = x_of & y_uf
            hist2d[-1, 0] += np.sum(weights[mask]) if weights is not None else np.sum(mask)

            # Overflow X with overflow Y (esquina [-1,-1])
            mask = x_of & y_of
            hist2d[-1, -1] += np.sum(weights[mask]) if weights is not None else np.sum(mask)

        # Underflow en Y (primer bin en Y)
        if include_underflow:
            # Underflow Y con X en rango
            mask = y_uf & x_in
            if weights is not None:
                hist2d[central_slice[0], 0] += np.histogram(x[mask], bins=bins_x, weights=weights[mask])[0]
            else:
                hist2d[central_slice[0], 0] += np.histogram(x[mask], bins=bins_x)[0]

        # Overflow en Y (último bin en Y)
        if include_overflow:
            # Overflow Y con X en rango
            mask = y_of & x_in
            if weights is not None:
                hist2d[central_slice[0], -1] += np.histogram(x[mask], bins=bins_x, weights=weights[mask])[0]
            else:
                hist2d[central_slice[0], -1] += np.histogram(x[mask], bins=bins_x)[0]

    return hist2d

def get_lumi_text(year) -> str:
    """Generate appropriate luminosity label based on year"""
    
    lumi_map = {
        "2017": "41.5 fb$^{-1}$ (2017, 13 TeV)",
        "2018": "59.8 fb$^{-1}$ (2018, 13 TeV)",
        "2016": "16.8 fb$^{-1}$ (2016, 13 TeV)",
        "2016APV": "19.5 fb$^{-1}$ (2016, 13 TeV)"
    }
    
    return lumi_map.get(year, "")

def plot_2d_hist(
    hist2d,
    bins_x,
    bins_y,
    xlabel,
    ylabel,
    title="",
    logz=False,
    save_path=None,
    year="2018",
    show_binning=False,
    show_bin_values=True,
    bin_value_fmt=".1f",
    bin_value_size=8
):
    """
    Plotea un histograma 2D con estilo CMS, mostrando valores consistentes con get_binning_table
    y devolviendo los bins extendidos que incluyen underflow/overflow
    """
    plt.style.use(hep.style.CMS)
    
    bins_x = np.asarray(bins_x)
    bins_y = np.asarray(bins_y)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Calcular bin widths para extensión
    dx = bins_x[1] - bins_x[0] if len(bins_x) > 1 else 0
    dy = bins_y[1] - bins_y[0] if len(bins_y) > 1 else 0

    # Crear bins extendidos (igual que en get_binning_table)
    extended_bins_x = np.concatenate([[bins_x[0] - dx], bins_x, [bins_x[-1] + dx]]) if dx > 0 else bins_x
    extended_bins_y = np.concatenate([[bins_y[0] - dy], bins_y, [bins_y[-1] + dy]]) if dy > 0 else bins_y

    if show_binning:
        print(f"Extended X bins: {extended_bins_x}")
        print(f"Extended Y bins: {extended_bins_y}")

    X, Y = np.meshgrid(extended_bins_x, extended_bins_y)

    # Dibujar histograma con bins extendidos
    mesh = ax.pcolormesh(
        X, Y, hist2d.T,
        cmap='viridis',
        shading='auto',
        norm=LogNorm() if logz else None
    )

    # Establecer límites a los bins regulares
    ax.set_xlim(bins_x[0], bins_x[-1])
    ax.set_ylim(bins_y[0], bins_y[-1])
    
    ax.set_xticks(bins_x)
    ax.set_yticks(bins_y)
    
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))

    # Añadir valores en los bins (coherente con get_binning_table)
    if show_bin_values:
        # Prepara el histograma visible (sumando underflow/overflow)
        hist_visible = hist2d.copy()
        nx = len(bins_x) - 1
        ny = len(bins_y) - 1
        
        if hist2d.shape[0] > nx and hist2d.shape[1] > ny:  # Si hay underflow/overflow
            # Sumar underflow/overflow (misma lógica que get_binning_table)
            hist_visible[1, 1:-1] += hist_visible[0, 1:-1]  # Underflow X
            hist_visible[nx, 1:-1] += hist_visible[nx+1, 1:-1]  # Overflow X
            hist_visible[1:-1, 1] += hist_visible[1:-1, 0]  # Underflow Y
            hist_visible[1:-1, ny] += hist_visible[1:-1, ny+1]  # Overflow Y
            # Esquinas
            hist_visible[1, 1] += hist_visible[0, 0]
            hist_visible[nx, ny] += hist_visible[nx+1, ny+1]
            hist_visible[1, ny] += hist_visible[0, ny+1]
            hist_visible[nx, 1] += hist_visible[nx+1, 0]

        # Centros de bins regulares
        bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
        bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2

        # Mostrar valores (solo para bins regulares)
        for i in range(nx):
            for j in range(ny):
                value = hist_visible[i+1, j+1] if (hist2d.shape[0] > nx and hist2d.shape[1] > ny) else hist_visible[i, j]
                if value == 0:
                    continue
                    
                x_pos = bin_centers_x[i]
                y_pos = bin_centers_y[j]
                
                rgba = mesh.cmap(mesh.norm(value))
                brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
                text_color = 'white' if brightness < 0.5 else 'black'
                
                ax.text(
                    x_pos, y_pos,
                    f"{value:{bin_value_fmt}}",
                    ha='center', va='center',
                    color=text_color, fontsize=bin_value_size,
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor=(*rgba[:3], 0.7),
                        edgecolor='none',
                        alpha=0.7
                    )
                )

    lumi_text = get_lumi_text(year=year)
    hep.cms.lumitext(lumi_text, fontsize=14, ax=ax)
    hep.cms.text("Preliminary", loc=0.0, fontsize=16, ax=ax)
    
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Events", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title:
        ax.set_title(title, fontsize=13, pad=15)

    ax.tick_params(labelsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado en: {save_path}")

    plt.show()

    return extended_bins_x, extended_bins_y  # Devuelve los bins extendidos




def get_group_hist2d(
    group_name: str,
    grouped_samples: dict,
    pkls: dict,
    norms: dict,
    feature_x: str,
    feature_y: str,
    bins_x: np.ndarray,
    bins_y: np.ndarray,
    weights_key: str = "weights",
    include_overflow: bool = True,
    include_underflow: bool = True,
    verbose: bool = False
):
    """
    Retorna un histograma combinado para un grupo o el total (excluyendo 'data').
    
    group_name: puede ser 'tt', 'wj', 'total', etc.
    """
    # Obtener lista de samples
    if group_name == "total":
        samples = [
            sample
            for g, lst in grouped_samples.items()
            if g != "data"  # Excluye datos reales
            for sample in lst
        ]
    elif group_name in grouped_samples:
        samples = grouped_samples[group_name]
    else:
        raise ValueError(f"Grupo '{group_name}' no reconocido.")

    # Histograma acumulado
    hist_total = np.zeros((
        len(bins_x)-1 + (1 if include_overflow else 0) + (1 if include_underflow else 0),
        len(bins_y)-1 + (1 if include_overflow else 0) + (1 if include_underflow else 0)
    ))

    for sample in samples:
        if sample not in pkls or sample not in norms:
            if verbose:
                print(f"⚠️ Sample '{sample}' no está en pkls o norms, se omite.")
            continue
            
        if any(key.endswith("_2016APV") for key in pkls.keys()):
            expected_key = sample.rsplit("_", 1)[0]
        else:
            expected_key = sample


        sample_dict = pkls[sample][expected_key]
        norm = norms[sample]

        hist = get_2d_hist_single_sample(
            sample_dict=sample_dict,
            feature_x=feature_x,
            feature_y=feature_y,
            bins_x=bins_x,
            bins_y=bins_y,
            weights_key=weights_key,
            norm_factor=norm,
            include_overflow=include_overflow,
            include_underflow=include_underflow,
            verbose=verbose
        )

        hist_total += hist

    return hist_total


def get_binning_table(hist2d, bins_x, bins_y, x_name="x", y_name="y"):
    """
    Genera una tabla con los bines centrales del histograma 2D,
    sumando correctamente underflow y overflow a los extremos correspondientes.
    Omite bins con conteo cero. Agrega una fila final con la suma total.

    Args:
        hist2d (np.ndarray): histograma 2D con underflow/overflow
        bins_x (np.ndarray): bordes del eje X
        bins_y (np.ndarray): bordes del eje Y
        x_name (str): nombre personalizado para el eje X
        y_name (str): nombre personalizado para el eje Y

    Returns:
        pd.DataFrame: tabla con columnas personalizadas y eventos
    """
    nx = len(bins_x) - 1
    ny = len(bins_y) - 1
    hist = hist2d.copy()

    # Under/overflow en X
    hist[1, 1:-1] += hist[0, 1:-1]
    hist[nx, 1:-1] += hist[nx+1, 1:-1]

    # Under/overflow en Y
    hist[1:-1, 1] += hist[1:-1, 0]
    hist[1:-1, ny] += hist[1:-1, ny+1]

    # Esquinas
    hist[1, 1]     += hist[0, 0]
    hist[nx, ny]   += hist[nx+1, ny+1]
    hist[1, ny]    += hist[0, ny+1]
    hist[nx, 1]    += hist[nx+1, 0]

    data = []
    total = 0.0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            events = hist[i, j]
            if events == 0:
                continue
            x_bin = f"[{bins_x[i-1]}, {bins_x[i]})"
            y_bin = f"[{bins_y[j-1]}, {bins_y[j]})"
            data.append({
                x_name: x_bin,
                y_name: y_bin,
                "events": round(events, 2)
            })
            total += events

    data.append({
        x_name: "Total",
        y_name: "Total",
        "events": round(total, 2)
    })

    return pd.DataFrame(data)



