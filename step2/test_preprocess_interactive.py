import numpy as np
import pylidc as pl
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # Importa o widget do slider
import logging

# Assumimos que o script final se chama 'preprocess.py'
try:
    from preprocess import (resample_image, create_lung_mask, 
                              HU_WINDOW_MIN, HU_WINDOW_MAX, TARGET_SPACING)
except ImportError:
    print("ERRO: Certifique-se de que o script de pré-processamento final ('preprocess.py') está no mesmo diretório.")
    exit()

# --- Configurações do Teste ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Troque o UID para testar diferentes exames
SERIES_UID_TO_TEST = "1.3.6.1.4.1.14519.5.2.1.6279.6001.195913706607582347421429908613"

def interactive_volume_viewer(image_array, mask_array=None, title_prefix=""):
    """
    Cria um visualizador interativo para um volume 3D com um slider.
    Permite sobrepor uma máscara opcional.
    """
    # Define as dimensões da figura e dos eixos
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.15) # Deixa espaço para o slider

    initial_slice = image_array.shape[0] // 2
    
    # Exibe a imagem inicial
    im = ax.imshow(image_array[initial_slice], cmap='gray')
    ax.axis('off')

    # Exibe a máscara inicial, se fornecida
    if mask_array is not None:
        im_mask = ax.imshow(np.ma.masked_where(mask_array[initial_slice] == 0, mask_array[initial_slice]),
                            cmap='Reds', alpha=0.4)
    
    ax.set_title(f"{title_prefix}\nSlice: {initial_slice}", fontsize=14)

    # Cria o eixo para o slider
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03]) # [left, bottom, width, height]
    
    # Cria o widget do Slider
    slider = Slider(
        ax=ax_slider,
        label='Slice',
        valmin=0,
        valmax=image_array.shape[0] - 1,
        valinit=initial_slice,
        valstep=1 # Garante valores inteiros
    )

    def update(val):
        """Função chamada toda vez que o slider é movido."""
        slice_idx = int(slider.val)
        
        # Atualiza os dados da imagem
        im.set_data(image_array[slice_idx])
        
        # Atualiza os dados da máscara, se existir
        if mask_array is not None:
            im_mask.set_data(np.ma.masked_where(mask_array[slice_idx] == 0, mask_array[slice_idx]))
            
        # Atualiza o título
        ax.set_title(f"{title_prefix}\nSlice: {slice_idx}", fontsize=14)
        
        # Redesenha a figura
        fig.canvas.draw_idle()

    # Conecta a função de atualização ao evento "on_changed" do slider
    slider.on_changed(update)

    plt.show()


def run_visual_tests(series_uid):
    """Executa um pipeline de teste visual para um único exame."""
    logging.info(f"--- Iniciando Teste Visual Interativo para o Exame: {series_uid} ---")
    
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
    if scan is None: logging.error("Exame não encontrado."); return
    
    # Carregamento e pré-processamento inicial
    vol_array_yxz = scan.to_volume(verbose=False)
    original_array = np.transpose(vol_array_yxz, (2, 0, 1))
    spacing_xyz = (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_thickness))
    original_sitk = sitk.GetImageFromArray(original_array)
    original_sitk.SetSpacing(spacing_xyz)
    
    resampled_sitk = resample_image(original_sitk, TARGET_SPACING)
    resampled_array = sitk.GetArrayFromImage(resampled_sitk)
    hu_windowed_array = np.clip(resampled_array, HU_WINDOW_MIN, HU_WINDOW_MAX)
    
    # Geração da máscara
    logging.info("Gerando máscara pulmonar para verificação...")
    lung_mask = create_lung_mask(hu_windowed_array)

    # --- CHAMADA PARA O VISUALIZADOR INTERATIVO ---
    # Este é o principal ponto de validação.
    if lung_mask.sum() > 0:
        interactive_volume_viewer(hu_windowed_array, lung_mask, title_prefix="Máscara Pulmonar Sobreposta (Interativo)")
    else:
        logging.warning("Máscara vazia. Não é possível mostrar o visualizador interativo.")
        interactive_volume_viewer(hu_windowed_array, title_prefix="Imagem sem Máscara (Falha na Geração)")

    logging.info("--- Teste Visual Interativo Concluído ---")


if __name__ == "__main__":
    run_visual_tests(SERIES_UID_TO_TEST)