import pylidc as pl
import pandas as pd
from tqdm import tqdm
import logging
import os

# Configuração do logging para registrar informações e erros
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ingestion_and_qc(output_csv_path: str = 'lidc_validated_patients_v2.csv'):
    """
    Executa a Etapa 1 (Refinada): Ingestão e Controle de Qualidade do LIDC-IDRI.

    Esta versão refinada:
    1. Consulta todos os exames de CT disponíveis.
    2. Usa `getattr` para capturar metadados de forma segura, preenchendo com 'None' 
       se um atributo (como 'study_date') não existir, em vez de pular o exame.
    3. Descarta um exame apenas se o 'patient_id' (crítico para o pipeline) estiver ausente.
    4. Gera um arquivo CSV mais completo, mostrando quais dados estão faltando.
    """
    logging.info("Iniciando Etapa 1 (Refinada): Ingestão e Controle de Qualidade.")

    all_scans_query = pl.query(pl.Scan)
    total_scans_found = all_scans_query.count()
    logging.info(f"Encontrados {total_scans_found} exames de CT no LIDC-IDRI.")

    validated_scans_metadata = []
    scans_with_missing_data = 0
    
    patients = {}

    for scan in tqdm(all_scans_query, desc="Validando exames e agrupando por paciente"):
        try:
            # O 'patient_id' é essencial, se falhar, pulamos o exame.
            patient_id = scan.patient_id
            
            # ALTERADO: Acessamos os outros metadados de forma segura usando getattr.
            # Se o atributo não existir, o valor será 'None'.
            scan_date = getattr(scan, 'study_date', None)
            slice_thickness = getattr(scan, 'slice_thickness', None)
            pixel_spacing = getattr(scan, 'pixel_spacing', None)

            # Contabiliza quantos exames possuem algum dado faltante (mas ainda serão incluídos)
            if not all([scan_date, slice_thickness, pixel_spacing]):
                scans_with_missing_data += 1
                
            scan_info = {
                'patient_id': patient_id,
                'scan_id': scan.id,
                'series_uid': scan.series_instance_uid, # Adicionado para melhor rastreabilidade
                'study_date': scan_date,
                'slice_thickness': slice_thickness,
                'pixel_spacing': pixel_spacing,
                'num_annotations': len(scan.annotations)
            }
            validated_scans_metadata.append(scan_info)
            
            if patient_id not in patients:
                patients[patient_id] = []
            patients[patient_id].append(scan)

        except AttributeError as e:
            # Este bloco agora só vai capturar erros se 'patient_id' ou outro atributo
            # não acessado com getattr falhar.
            logging.error(f"Exame com ID '{scan.series_instance_uid}' pulado por erro crítico. Erro: {e}")
        except Exception as e:
            logging.error(f"Erro inesperado ao processar o exame '{scan.series_instance_uid}'. Erro: {e}")


    logging.info("-" * 50)
    logging.info("Controle de Qualidade Finalizado.")
    logging.info(f"Total de exames processados: {total_scans_found}")
    logging.info(f"Total de exames incluídos na lista final: {len(validated_scans_metadata)}")
    logging.info(f"   - Desses, {scans_with_missing_data} possuem algum metadado não-crítico ausente.")
    logging.info(f"Total de pacientes únicos encontrados: {len(patients)}")

    if validated_scans_metadata:
        df_validated = pd.DataFrame(validated_scans_metadata)
        df_validated.to_csv(output_csv_path, index=False)
        logging.info(f"Relatório de qualidade salvo com sucesso em: '{output_csv_path}'")
    else:
        logging.error("Nenhum exame validado foi encontrado. O arquivo de saída não foi gerado.")

    return patients, pd.DataFrame(validated_scans_metadata)


if __name__ == '__main__':
    valid_patients_dict, validated_scans_df = run_ingestion_and_qc()
    
    print("\n--- Resumo do Relatório de Qualidade (v2) ---")
    if not validated_scans_df.empty:
        print(f"O arquivo '{os.path.basename(validated_scans_df.attrs.get('name', 'lidc_validated_patients_v2.csv'))}' foi gerado.")
        print("Primeiras 5 linhas do relatório:")
        print(validated_scans_df.head().to_string())
    else:
        print("Nenhum dado foi processado com sucesso.")