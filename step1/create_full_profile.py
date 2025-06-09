import pylidc as pl
import pandas as pd
from tqdm import tqdm
import logging
from collections import Counter

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURAÇÃO DAS CARACTERÍSTICAS A SEREM PROCESSADAS ---
# Dicionário que mapeia o nome do atributo ao seus possíveis valores.
# Facilita a adição ou remoção de características no futuro.
ATTRIBUTES_TO_COUNT = {
    'subtlety': list(range(1, 6)),
    'internalStructure': list(range(1, 5)),
    'calcification': [1, 2, 3, 4, 5, 6], # O valor 5 não existe, mas incluímos para um range completo visual
    'sphericity': list(range(1, 6)),
    'margin': list(range(1, 6)),
    'lobulation': list(range(1, 6)),
    'spiculation': list(range(1, 6)),
    'texture': list(range(1, 6)),
    'malignancy': list(range(1, 6))
}

def create_full_dataset_profile(input_csv: str, output_csv: str):
    """
    Cria um perfil completo do dataset, enriquecendo-o com a contagem de todas as
    características das anotações (subtlety, texture, malignancy, etc.).

    Args:
        input_csv (str): Caminho para o CSV gerado pelo script de QC (v2).
        output_csv (str): Caminho para salvar o CSV final com o perfil completo.
    """
    logging.info(f"Lendo o arquivo de metadados base: '{input_csv}'")
    try:
        df_base = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Arquivo de entrada não encontrado: {input_csv}. Execute o 'ingestion_qc_v2.py' primeiro.")
        return

    logging.info(f"Iniciando perfilamento completo para {len(ATTRIBUTES_TO_COUNT)} características...")
    
    all_profiles_data = []

    for series_uid in tqdm(df_base['series_uid'], desc="Gerando perfil dos exames"):
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).first()
        
        # Dicionário para armazenar o perfil completo deste exame
        scan_profile = {'series_uid': series_uid}

        if scan is None or len(scan.annotations) == 0:
            # Se não houver anotações, todas as contagens são zero
            for attr, values in ATTRIBUTES_TO_COUNT.items():
                for val in values:
                    scan_profile[f'{attr}_{val}'] = 0
        else:
            # Itera sobre cada característica que queremos contar
            for attr, values in ATTRIBUTES_TO_COUNT.items():
                # Extrai os valores da característica de todas as anotações do exame
                attr_values_in_scan = [getattr(ann, attr) for ann in scan.annotations]
                # Conta a frequência de cada valor
                value_counts = Counter(attr_values_in_scan)
                # Preenche o perfil do exame com as contagens
                for val in values:
                    scan_profile[f'{attr}_{val}'] = value_counts.get(val, 0)
        
        all_profiles_data.append(scan_profile)

    # Cria um DataFrame com todos os perfis
    df_profiles = pd.DataFrame(all_profiles_data)

    # Funde o DataFrame original com o de perfis
    logging.info("Fundindo os perfis com o dataset original.")
    df_final = pd.merge(df_base, df_profiles, on='series_uid')
    
    df_final.to_csv(output_csv, index=False)
    logging.info(f"Dataset final com perfil completo salvo em: '{output_csv}'")
    return df_final

if __name__ == '__main__':
    input_file = 'lidc_validated_patients_v2.csv'
    output_file = 'lidc_dataset_final_profile.csv'
    
    final_df = create_full_dataset_profile(input_csv=input_file, output_csv=output_file)
    
    if final_df is not None:
        print("\n--- Perfilamento Completo Concluído ---")
        print(f"Dataset final salvo em '{output_file}' com {final_df.shape[1]} colunas.")
        print("\nAmostra do dataset final (primeiras e últimas colunas):")
        # Exibe uma amostra para dar uma ideia da estrutura
        cols_to_show = list(final_df.columns[:6]) + list(final_df.columns[-6:])
        print(final_df[cols_to_show].head().to_string())