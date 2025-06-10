import numpy as np
import pylidc as pl
import SimpleITK as sitk
import matplotlib.pyplot as plt
import logging

# Tenta importar as funções da versão final do script de pré-processamento.
# Assumimos que o arquivo foi salvo como `preprocess.py`.
try:
    from preprocess import (resample_image, create_lung_mask, 
                              HU_WINDOW_MIN, HU_WINDOW_MAX, TARGET_SPACING)
except ImportError:
    print("ERRO: Certifique-se de que o script de pré-processamento final ('preprocess.py') está no mesmo diretório.")
    exit()

# --- Configurações do Teste ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# UID de um exame para teste. Você pode trocar por qualquer outro do seu CSV.
SERIES_UID_TO_TEST = "1.3.6.1.4.1.14519.5.2.1.6279.6001.102133688497886810253331438797"


def plot_slice(image_array, title, slice_index=None):
    """Plota o corte central de um volume 3D."""
    if slice_index is None:
        slice_index = image_array.shape[0] // 2
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array[slice_index], cmap='gray')
    plt.title(f"{title}\nSlice: {slice_index}", fontsize=14)
    plt.axis('off')
    plt.show()

def plot_slice_with_mask(image_array, mask_array, title, slice_index=None):
    """Plota um corte de imagem com uma máscara sobreposta."""
    if slice_index is None:
        slice_index = image_array.shape[0] // 2

    plt.figure(figsize=(10, 10))
    plt.imshow(image_array[slice_index], cmap='gray')
    plt.imshow(np.ma.masked_where(mask_array[slice_index] == 0, mask_array[slice_index]),
               cmap='Reds', alpha=0.4)
    plt.title(f"{title}\nSlice: {slice_index}", fontsize=14)
    plt.axis('off')
    plt.show()

def plot_histograms(before_array, after_array, title_before, title_after):
    """Plota histogramas de intensidade de voxel antes e depois da normalização."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(before_array.flatten(), bins=100, color='blue')
    axes[0].set_title(title_before, fontsize=14); axes[0].set_ylabel("Frequência"); axes[0].set_xlabel("Intensidade (HU)")
    axes[1].hist(after_array.flatten(), bins=100, color='green')
    axes[1].set_title(title_after, fontsize=14); axes[1].set_xlabel("Intensidade (Z-score)")
    fig.suptitle("Comparação de Histogramas de Intensidade", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


def run_visual_tests(series_uid):
    """Executa um pipeline de teste visual para um único exame."""
    logging.info(f"--- Iniciando Teste Visual (Final) para o Exame: {series_uid} ---")
    
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None: logging.error("Exame não encontrado."); return
    
    # --- LÓGICA DE CARREGAMENTO CORRETA (ESPELHANDO preprocess.py) ---
    # 1. Carregar o volume com o método correto `to_volume()`
    vol_array_yxz = scan.to_volume(verbose=False)  # Ordem de eixos: (y, x, z)

    # 2. Transpor para a convenção padrão (z, y, x)
    original_array = np.transpose(vol_array_yxz, (2, 0, 1))
    
    # 3. Criar objeto SimpleITK para obter metadados e passar para as próximas funções
    spacing_xyz = (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_thickness))
    original_sitk = sitk.GetImageFromArray(original_array)
    original_sitk.SetSpacing(spacing_xyz)
    # -------------------------------------------------------------------

    logging.info(f"Original - Shape: {original_array.shape}, Spacing: {np.round(original_sitk.GetSpacing(), 2)}")
    plot_slice(original_array, "1. Imagem Original (HU, eixos corrigidos)")
    
    # Daqui em diante, as funções de teste são as mesmas, pois recebem os dados já no formato correto
    resampled_sitk = resample_image(original_sitk, TARGET_SPACING)
    resampled_array = sitk.GetArrayFromImage(resampled_sitk)
    logging.info(f"Reamostrado - Shape: {resampled_array.shape}, Spacing: {np.round(resampled_sitk.GetSpacing(), 2)}")
    plot_slice(resampled_array, "2. Imagem Reamostrada (1x1x1 mm)")
    
    hu_windowed_array = np.clip(resampled_array, HU_WINDOW_MIN, HU_WINDOW_MAX)
    plot_slice(hu_windowed_array, f"3. Janela de HU Aplicada [{HU_WINDOW_MIN}, {HU_WINDOW_MAX}]")
    
    logging.info("Gerando máscara pulmonar para verificação...")
    lung_mask = create_lung_mask(hu_windowed_array)
    plot_slice_with_mask(hu_windowed_array, lung_mask, "4. Máscara Pulmonar Sobreposta")
    
    coords = np.argwhere(lung_mask)
    if coords.shape[0] > 0:
        z_min, y_min, x_min = coords.min(axis=0); z_max, y_max, x_max = coords.max(axis=0)
        cropped_array = hu_windowed_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        logging.info(f"Crop - Shape Final: {cropped_array.shape}")
        plot_slice(cropped_array, "5. Imagem Recortada (Lung Crop)")
    else:
        logging.warning("Máscara vazia. O crop não foi aplicado.")
        cropped_array = hu_windowed_array

    mean, std = np.mean(cropped_array), np.std(cropped_array)
    normalized_array = (cropped_array - mean) / std if std > 0 else cropped_array - mean
    plot_histograms(cropped_array, normalized_array, "Antes da Normalização (HU)", "Depois da Normalização (Z-score)")
    plot_slice(normalized_array, "6. Imagem Final Normalizada")

    logging.info("--- Teste Visual Concluído ---")


if __name__ == "__main__":
    run_visual_tests(SERIES_UID_TO_TEST)