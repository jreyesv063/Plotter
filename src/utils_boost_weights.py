import os
import json
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt


class WeightPlotter:
    def __init__(self, year, weight_case, com=13):
        """
        Inicializa el objeto de plotting.
        """
        self.year = year
        self.weight_case = weight_case
        self.com = com

        # Mapas de etiquetas y títulos
        self.lumi = {
            "2016APV": 19.5,
            "2016": 16.8,
            "2017": 41.5,
            "2018": 59.8
        }

        self.weight_map = {
            "ISR_Zmumu_weight_FxFx": f"ISR_weight_{year}_UL",
            "top_boost_tau": "top_boost_weight"
        }

        self.x_label = {
            "top_boost_tau": r'$S_{T}(\tau_{h}, HT, b\text{-jets}, p_{T}^{miss})$ [GeV]',
            "ISR_Zmumu_weight_FxFx": r'$p_{T}(\mu\mu)$ [GeV]'
        }

        self.y_label = {
            "top_boost_tau": r'$N_{\mathrm{jets}}$',
            "ISR_Zmumu_weight_FxFx": r'$N_{\mathrm{jets}}$'
        }

        self.title = {
            "ISR_Zmumu_weight_FxFx": "ISR weights",
            "top_boost_tau":  r"$t\bar{t}$ boost weights"
        }

    def _process_edges(self, edges, delta):
        """
        Convierte edges a float y expande infinitos.
        """
        processed = []
        for e in edges:
            if str(e).lower() in ["inf", "infinity"]:
                processed.append(processed[-1] + delta)
            else:
                processed.append(float(e))
        return np.array(processed, dtype=float)

    def load_json(self, ruta_json, key="nominal"):
        """
        Carga un JSON de correctionlib y devuelve (x_edges, y_edges, weights).
        """
        with open(ruta_json, "r") as f:
            data = json.load(f)

        for corr in data["corrections"]:
            if corr["name"] == self.weight_map[self.weight_case]:
                for entry in corr["data"]["content"]:
                    if entry["key"] == key:
                        edges_x, edges_y = entry["value"]["edges"]

                        if self.weight_case == "top_boost_tau":
                            x_edges = self._process_edges(edges_x, 100)   # ST
                            y_edges = self._process_edges(edges_y, 1)     # njets
                            weights = np.array(entry["value"]["content"]).reshape(len(y_edges)-1, len(x_edges)-1)

                        elif self.weight_case == "ISR_Zmumu_weight_FxFx":
                            # Ejes invertidos
                            y_edges = self._process_edges(edges_x, 1)     # njets
                            x_edges = self._process_edges(edges_y, 100)   # ST
                            weights = np.array(entry["value"]["content"]).reshape(len(y_edges)-1, len(x_edges)-1)

                        return x_edges, y_edges, weights

        raise ValueError(f"No se encontró la clave {key} en el JSON")

    def plot(self, json_path, output_file=None, key="nominal"):
        """
        Genera y guarda el heatmap.
        """
        x_edges, y_edges, weights = self.load_json(json_path, key)

        plt.figure(figsize=(16, 12)) 
        c = plt.pcolormesh(x_edges, y_edges, weights, shading='auto', cmap='viridis')
        cb = plt.colorbar(c, label='Weights')
        cb.ax.tick_params(labelsize=16) 
        cb.set_label("Weights", fontsize=20)

        # Anotar valores en cada celda
        for i in range(len(y_edges) - 1):
            for j in range(len(x_edges) - 1):
                value = weights[i, j]
                if np.isnan(value):
                    continue
                color = "white" if value < 0.9 else "black"
                plt.text((x_edges[j] + x_edges[j+1]) / 2,
                         (y_edges[i] + y_edges[i+1]) / 2,
                         f"{value:.2f}", ha='center', va='center',
                         fontsize=12, fontweight="bold", color=color)

        # Etiquetas de ejes
        plt.xlabel(self.x_label[self.weight_case], fontsize=20)
        plt.ylabel(self.y_label[self.weight_case], fontsize=20)
        plt.xticks(x_edges, rotation=45, fontsize=14)
        plt.yticks(y_edges, fontsize=14)
        plt.grid(False)

        # CMS label
        hep.cms.label(
            "Preliminary", 
            data=True, 
            lumi=self.lumi[self.year], 
            year=self.year, 
            com=self.com, 
            loc=0,
            fontsize=23
        )

        # Título
        plt.title(self.title[self.weight_case], fontsize=20,  pad=35)

        plt.tight_layout()

        # Guardar
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # <--- CREA LA CARPETA
            plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=300)
            print(f"✅ Gráfico guardado en {output_file}")

        plt.show()


