import pandas as pd
import pylidc as pl
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, ball, remove_small_objects, convex_hull_image
from skimage.filters import threshold_otsu
from scipy.spatial.distance import euclidean
import scipy.ndimage as ndi


# --- Configurações ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
OUTPUT_DIR = Path("./preprocessed_data")
NIFTI_DIR = OUTPUT_DIR / "nifti_volumes"
CROP_INFO_DIR = OUTPUT_DIR / "crop_info"
NIFTI_DIR.mkdir(parents=True, exist_ok=True)
CROP_INFO_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = "step1/lidc_dataset_final_profile.csv"
TARGET_SPACING = (1.0, 1.0, 1.0)
HU_WINDOW_MIN = -1000
HU_WINDOW_MAX = 400
CROP_PADDING = 10

def resample_image(sitk_image: sitk.Image, target_spacing: tuple) -> sitk.Image:
    original_spacing = sitk_image.GetSpacing(); original_size = sitk_image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter(); resampler.SetOutputSpacing(target_spacing); resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection()); resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform()); resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(sitk_image)

def denoise_gaussian(volume: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """
    Filtro Gaussiano 3-D leve. sigma=0.6 (~1 voxel FWHM) preserva bordas
    finas e remove ruído de alta frequência / artefato de metal fraco.
    """
    return ndi.gaussian_filter(volume, sigma=sigma)

def adaptive_air_threshold(slice_2d: np.ndarray,
                           lo: int = -1000, hi: int = 0,
                           t_min: int = -500, t_max: int = -250,
                           default: int = -320) -> int:
    """
    Calcula Otsu só dentro da faixa lo..hi.
    Clampa o resultado entre t_min e t_max para evitar outliers.
    """
    mask = (slice_2d >= lo) & (slice_2d <= hi)
    data = slice_2d[mask]
    if data.size < 50:      # slice vazia ou fora do tórax
        return default
    t = threshold_otsu(data)
    return int(np.clip(t, t_min, t_max))

def hu_histogram(vals: np.ndarray,
                 bins: int = 64,
                 hu_min: int = -1000,
                 hu_max: int = 0) -> np.ndarray:
    h, _ = np.histogram(vals,
                        bins=bins,
                        range=(hu_min, hu_max),
                        density=False)
    h = h.astype(np.float32) + 1e-6
    return h / h.sum()

def cos_hist(h1, h2):
    return float(np.dot(h1, h2) / (np.linalg.norm(h1)*np.linalg.norm(h2) + 1e-8))


# def create_lung_mask(vol: np.ndarray,
#                      min_blob_area_px: int = 600,
#                      prop_kernel_xy: int = 5) -> np.ndarray:
#     """
#     Gera máscara binária (0/1) dos dois pulmões em um CT torácico.
#     Estratégia:
#       1. Segmentar ar interno (HU < hu_air) slice-a-slice,
#          removendo ar externo (conectado à borda).
#       2. Determinar midline (coluna central do corpo) em cada slice.
#       3. Manter o MAIOR componente de ar em cada lado (E/D).
#       4. Aplicar retro-alimentação: se faltar pulmão num slice,
#          projetar a máscara do slice adjacente e re-intersectar.
#       5. Conectar 3-D, pegar 2 maiores comps e *closing* 3-D.
#     """
#     n_z, n_y, n_x = vol.shape
    
#     # ---------- THRESHOLD ADAPTATIVO ----------
#     thresh = np.array([adaptive_air_threshold(vol[z]) for z in range(n_z)], dtype=np.int16)
#     # suaviza variações abruptas (janela de 5)
#     thresh = ndi.median_filter(thresh, size=5)

#     lung_stack = np.zeros_like(vol, dtype=bool)
#     struct_xy = ndi.generate_binary_structure(2, 1)
#     prev_left = prev_right = None

#     for z in range(n_z):
#         sl = vol[z]
#         # ---------- 1. ar interno ----------
#         hu_air = thresh[z]
#         air = sl < hu_air
#         # remove ar que toca a borda
#         air_no_border = air.copy()
#         air_no_border[ndi.binary_fill_holes(~air)[0]] = False  # garante contorno fechado
#         border_labels = label(air_no_border)
#         border_touch = np.unique(np.concatenate([border_labels[0, :],
#                                                  border_labels[-1, :],
#                                                  border_labels[:, 0],
#                                                  border_labels[:, -1]]))
#         for lab in border_touch:
#             air_no_border[border_labels == lab] = False
#         # ---------- 2. corpo & midline ----------
#         body = sl > hu_air
#         if body.sum() == 0:
#             continue
#         cols = np.where(body.sum(0) > 0)[0]
#         if len(cols) < 4:  # slice fora do tórax
#             continue
#         midline = int(np.median(cols))

#         # ---------- 3. blobs por lado ----------
#         labels = label(air_no_border)
#         regions = [r for r in regionprops(labels) if r.area >= min_blob_area_px]

#         left_blob = right_blob = None
#         for r in regions:
#             (cy, cx) = r.centroid
#             if cx < midline:
#                 if left_blob is None or r.area > left_blob.area:
#                     left_blob = r
#             else:
#                 if right_blob is None or r.area > right_blob.area:
#                     right_blob = r

#         # ---------- 4. retro-alimentação ----------
#         # se não achar blob em um lado mas existia no slice anterior,
#         # dilata o anterior e usa a intersecção com ar_no_border
#         if left_blob is None and prev_left is not None:
#             dil = ndi.binary_dilation(prev_left, structure=ndi.generate_binary_structure(2, 1),
#                                        iterations=prop_kernel_xy)
#             left_mask = np.logical_and(dil, air_no_border)
#             if left_mask.sum() > 0:
#                 left_blob = regionprops(label(left_mask))[0]

#         if right_blob is None and prev_right is not None:
#             dil = ndi.binary_dilation(prev_right, structure=struct_xy, iterations=prop_kernel_xy)
#             right_mask = np.logical_and(dil, air_no_border)
#             if right_mask.sum() > 0:
#                 right_blob = regionprops(label(right_mask))[0]

#         # grava máscara e guarda para o próximo slice
#         if left_blob is not None:
#             lung_stack[z, labels == left_blob.label] = True
#             prev_left = labels == left_blob.label
#         else:
#             prev_left = None

#         if right_blob is not None:
#             lung_stack[z, labels == right_blob.label] = True
#             prev_right = labels == right_blob.label
#         else:
#             prev_right = None

#     # ---------- 5. pós-processamento 3-D ----------
#     # remove objetos minúsculos (traqueia, vazios)
#     lung_stack = remove_small_objects(lung_stack, min_size=3_000)
#     labels3d = label(lung_stack)
#     regions3d = sorted(regionprops(labels3d), key=lambda r: r.area, reverse=True)[:2]

#     final_mask = np.zeros_like(lung_stack)
#     for r in regions3d:
#         final_mask[labels3d == r.label] = True

#     # fechamento morfológico 3-D para eliminar fissuras pequenas
#     final_mask = binary_closing(final_mask, ball(3))

#     return final_mask.astype(np.int8)

# def create_lung_mask(vol: np.ndarray,
#                      alpha: float = 0.60,           # peso da retro-alimentação
#                      thr_final: float = 0.50,       # limiar de decisão
#                      w_size: float = 0.4,
#                      w_center: float = 0.4,
#                      w_shape: float = 0.2,
#                      min_blob_area_px: int = 300) -> np.ndarray:
#     """
#     Retorna máscara (0/1) dos pulmões usando plausibilidade contínua.
#     alpha       – fração da nota herdada da fatia anterior
#     thr_final   – voxels com nota ≥ thr_final viram 1
#     w_*         – pesos dos critérios na nota inicial
#     """

#     n_z, n_y, n_x = vol.shape
#     diag = np.sqrt(n_y**2 + n_x**2)              # p/ normalizar distância ao centro

#     # -------- 1. PLAUSIBILIDADE INICIAL --------
#     plaus_init = np.zeros_like(vol, dtype=np.float32)

#     for z in range(n_z):
#         sl = vol[z]
#         hu_air = adaptive_air_threshold(sl)
#         air = sl < hu_air

#         # remove ar conectado à borda
#         air_nb = air.copy()
#         air_nb[ndi.binary_fill_holes(~air)[0]] = False
#         border_labels = label(air_nb)
#         border_touch = np.unique(np.concatenate([border_labels[0, :],
#                                                  border_labels[-1, :],
#                                                  border_labels[:, 0],
#                                                  border_labels[:, -1]]))
#         for lab in border_touch:
#             air_nb[border_labels == lab] = False

#         labels2d = label(air_nb)
#         regions = [r for r in regionprops(labels2d) if r.area >= min_blob_area_px]
#         if not regions:
#             continue

#         slice_area = n_y * n_x
#         center_y, center_x = n_y / 2.0, n_x / 2.0

#         for r in regions:
#             # --- critérios ---
#             size_score   = min(1.0, r.area / (0.25 * slice_area))        # 1 se ≥25% da área
#             cy, cx       = r.centroid
#             dist_center  = np.hypot(cy - center_y, cx - center_x)
#             center_score = 1.0 - dist_center / (diag / 2.0)
#             shape_score  = 1.0 - r.eccentricity          # 1 para círculo

#             score = (w_size*size_score +
#                      w_center*center_score +
#                      w_shape*shape_score) / (w_size + w_center + w_shape)

#             plaus_init[z, labels2d == r.label] = score.astype(np.float32)

#     # -------- 2. RETRO-ALIMENTAÇÃO VERTICAL --------
#     plaus_final = np.zeros_like(plaus_init, dtype=np.float32)
#     for z in range(n_z):
#         if z == 0:
#             plaus_final[z] = plaus_init[z]
#         else:
#             plaus_final[z] = plaus_init[z] + alpha * plaus_final[z-1]
#             plaus_final[z] = np.clip(plaus_final[z], 0.0, 1.0)

#     # -------- 3. LIMIAR + LIMPEZA 3-D --------
#     lung_mask = plaus_final >= thr_final
#     lung_mask = remove_small_objects(lung_mask, min_size=3_000)

#     # Mantemos os 2 maiores componentes 3-D (pulmões)
#     labels3d   = label(lung_mask)
#     regions3d  = sorted(regionprops(labels3d), key=lambda r: r.area, reverse=True)[:2]
#     final = np.zeros_like(lung_mask)
#     for r in regions3d:
#         final[labels3d == r.label] = True

#     final = binary_closing(final, ball(3))
#     return final.astype(np.int8)

def create_lung_mask(vol: np.ndarray,
                     alpha: float = 0.6,        # feedback weight
                     thr_final: float = 0.50,   # cutoff for mask
                     w_size: float = 0.25,
                     w_center: float = 0.25,
                     w_shape: float = 0.2,
                     w_sim: float = 0.30,
                     min_blob_area_px: int = 300) -> np.ndarray:
    """
    Mapa de plausibilidade + retro-alimentação com 4 critérios:
        • tamanho      (w_size)
        • centralidade (w_center)
        • circularidade/excentricidade (w_shape)
        • similaridade de histograma HU (w_sim, cosseno)
    """
    n_z, n_y, n_x = vol.shape
    diag = np.hypot(n_y, n_x)

    # 1. mapa de plausibilidade inicial
    plaus_init = np.zeros_like(vol, dtype=np.float32)
    hist_ref = None                                 # referência dinâmica

    for z in range(n_z):
        sl      = vol[z]
        hu_air  = adaptive_air_threshold(sl)
        air     = sl < hu_air

        air_nb  = air.copy()
        air_nb[ndi.binary_fill_holes(~air)[0]] = False
        border_lab = label(air_nb)
        for lab in np.unique(np.concatenate([border_lab[0, :],
                                             border_lab[-1, :],
                                             border_lab[:, 0],
                                             border_lab[:, -1]])):
            air_nb[border_lab == lab] = False

        labels  = label(air_nb)
        regions = [r for r in regionprops(labels) if r.area >= min_blob_area_px]
        if not regions:
            continue

        slice_area   = n_y * n_x
        cy_global, cx_global = n_y / 2.0, n_x / 2.0

        for r in regions:
            # --- critérios geométricos ---
            size_score   = min(1.0, r.area / (0.25 * slice_area))
            cy, cx       = r.centroid
            center_score = 1.0 - np.hypot(cy - cy_global,
                                          cx - cx_global) / (diag / 2.0)
            shape_score  = 1.0 - r.eccentricity

            # --- similaridade espectral ---
            hist_cand = hu_histogram(sl[labels == r.label])
            sim_score = cos_hist(hist_cand, hist_ref) if hist_ref is not None else 0.5

            score = (w_size*size_score +
                     w_center*center_score +
                     w_shape*shape_score +
                     w_sim*sim_score) / (w_size + w_center + w_shape + w_sim)

            plaus_init[z, labels == r.label] = score.astype(np.float32)

        # ---------- atualização da referência ----------
        # voxels com score alto (>0.6) nesta fatia servem de novo “pulmão típico”
        vox_idx = plaus_init[z] > 0.6
        if vox_idx.any():
            hist_ref = hu_histogram(sl[vox_idx])

    # 2. retro-alimentação vertical
    plaus_final = np.zeros_like(plaus_init, dtype=np.float32)
    for z in range(n_z):
        if z == 0:
            plaus_final[z] = plaus_init[z]
        else:
            plaus_final[z] = np.clip(plaus_init[z] + alpha * plaus_final[z-1], 0.0, 1.0)

    # 3. decisão final + limpeza
    lung_mask = plaus_final >= thr_final
    lung_mask = remove_small_objects(lung_mask, min_size=3_000)

    labels3d  = label(lung_mask)
    regions3d = sorted(regionprops(labels3d), key=lambda r: r.area, reverse=True)[:2]
    final = np.zeros_like(lung_mask)
    for r in regions3d:
        final[labels3d == r.label] = True

    return binary_closing(final, ball(3)).astype(np.int8)


def process_scan(series_uid: str):
    """Executa todo o pipeline de pré-processamento para um único exame (scan)."""
    try:
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
        if scan is None: logging.warning(f"Scan {series_uid} não encontrado. Pulando."); return

        vol_array_yxz = scan.to_volume(verbose=False)
        vol_array = np.transpose(vol_array_yxz, (2, 0, 1))

        spacing_xyz = (float(scan.pixel_spacing), float(scan.pixel_spacing), float(scan.slice_thickness))
        sitk_image = sitk.GetImageFromArray(vol_array)
        sitk_image.SetSpacing(spacing_xyz)
        
        resampled_sitk_image = resample_image(sitk_image, TARGET_SPACING)
        resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)
        hu_windowed_array = np.clip(resampled_array, HU_WINDOW_MIN, HU_WINDOW_MAX)
        denoised_array = denoise_gaussian(hu_windowed_array, sigma=0.6)
        
        # --- Chamando a função de máscara definitiva com lógica de centroide ---
        lung_mask = create_lung_mask(denoised_array)
        
        if lung_mask.sum() < 1000:
            logging.warning(f"Máscara pulmonar suspeita para {series_uid}. Pulando o crop.")
            cropped_array = hu_windowed_array; bbox = (0, 0, 0, *hu_windowed_array.shape)
        else:
            coords = np.argwhere(lung_mask)
            z_min, y_min, x_min = coords.min(axis=0); z_max, y_max, x_max = coords.max(axis=0)
            
            # --- Adicionando padding de segurança ao Bounding Box ---
            shape = hu_windowed_array.shape
            z_min = np.maximum(0, z_min - CROP_PADDING); y_min = np.maximum(0, y_min - CROP_PADDING); x_min = np.maximum(0, x_min - CROP_PADDING)
            z_max = np.minimum(shape[0], z_max + CROP_PADDING); y_max = np.minimum(shape[1], y_max + CROP_PADDING); x_max = np.minimum(shape[2], x_max + CROP_PADDING)

            bbox = (z_min, y_min, x_min, z_max, y_max, x_max)
            cropped_array = hu_windowed_array[z_min:z_max, y_min:y_max, x_min:x_max]
        
        mean, std = np.mean(cropped_array), np.std(cropped_array)
        normalized_array = (cropped_array - mean) / std if std > 0 else cropped_array - mean

        final_sitk_image = sitk.GetImageFromArray(normalized_array.astype(np.float32))
        final_sitk_image.SetSpacing(TARGET_SPACING)
        output_path = NIFTI_DIR / f"{series_uid}.nii.gz"
        sitk.WriteImage(final_sitk_image, str(output_path))
        
        bbox_path = CROP_INFO_DIR / f"{series_uid}_bbox.npy"
        np.save(bbox_path, np.array(bbox))

    except Exception as e:
        logging.error(f"Falha ao processar {series_uid}. Erro: {e}", exc_info=True)




if __name__ == "__main__":
    logging.info("--- Iniciando Etapa 2: Pré-processamento (v5 - Definitivo) ---")
    df = pd.read_csv(INPUT_CSV)
    
    for series_uid in tqdm(df['series_uid'], desc="Processando exames de CT"):
        process_scan(series_uid)
        
    logging.info("--- Etapa 2: Pré-processamento Concluída ---")