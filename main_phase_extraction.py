import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from src.msca.processing.processing_deflectometry import PMDProcessing

def load_images_from_directory(directory, prefix="cam0"):
    """
    Lê os arquivos .png de uma câmera específica de um diretório.
    Retorna uma lista de arrays numpy em escala de cinza.
    """
    files = sorted(glob.glob(os.path.join(directory, f'{prefix}*.png')))
    if len(files) == 0:
        print(f"Aviso: Nenhuma imagem encontrada em {directory} com o prefixo {prefix}")
        return []
    
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
    return imgs

def evaluate_quality(wrapped_phase, modulation, axis_name, unwrapped_phase=None):
    """
    Gera e salva mapas e gráficos de avaliação de qualidade do resultado.
    """
    # Salvar mapa de modulação (SNR) como imagem visual
    mod_norm = cv2.normalize(modulation, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"out/modulation_map_{axis_name}.png", mod_norm)
    
    # Se houver fase desembrulhada, gerar uma visualização em PNG também
    if unwrapped_phase is not None:
        p_min, p_max = np.nanmin(unwrapped_phase), np.nanmax(unwrapped_phase)
        if p_max > p_min:
            unwrapped_norm = ((unwrapped_phase - p_min) / (p_max - p_min) * 255)
            # Para onde for nan (fundo/ruído mascarado), coloca 0 (preto)
            unwrapped_norm = np.where(np.isnan(unwrapped_phase), 0, unwrapped_norm).astype(np.uint8)
            cv2.imwrite(f"out/unwrapped_phase_{axis_name}.png", unwrapped_norm)
            
    plt.figure(figsize=(15, 5))
    
    # 1. Histograma da Modulação
    plt.subplot(1, 3, 1)
    mod_values = modulation.ravel()
    plt.hist(mod_values[mod_values > 1e-3], bins=100, color='blue', alpha=0.7)
    plt.title(f"Modulação (SNR) - Eixo {axis_name}")
    plt.xlabel("Valor de Modulação")
    plt.ylabel("Frequência")
    plt.grid(True)
    
    # 2. Perfil 1D da Fase Embrulhada
    plt.subplot(1, 3, 2)
    H, W = wrapped_phase.shape
    if axis_name == 'X':
        profile_w = wrapped_phase[H // 2, :]
        plt.plot(profile_w, color='red', linewidth=1.5)
        plt.title(f"Fase Wrapped (Linha {H//2})")
        plt.xlabel("Pixels em X")
    else:
        profile_w = wrapped_phase[:, W // 2]
        plt.plot(profile_w, color='green', linewidth=1.5)
        plt.title(f"Fase Wrapped (Coluna {W//2})")
        plt.xlabel("Pixels em Y")
    plt.ylabel("Fase (radianos)")
    plt.grid(True)
    
    # 3. Perfil 1D da Fase Desembrulhada
    plt.subplot(1, 3, 3)
    if unwrapped_phase is not None:
        if axis_name == 'X':
            profile_u = unwrapped_phase[H // 2, :]
            plt.plot(profile_u, color='purple', linewidth=1.5)
            plt.title(f"Fase Unwrapped (Linha {H//2})")
            plt.xlabel("Pixels em X")
        else:
            profile_u = unwrapped_phase[:, W // 2]
            plt.plot(profile_u, color='orange', linewidth=1.5)
            plt.title(f"Fase Unwrapped (Coluna {W//2})")
            plt.xlabel("Pixels em Y")
        plt.ylabel("Fase absoluta (radianos)")
    else:
        plt.text(0.5, 0.5, 'Desembrulho não realizado', ha='center', va='center')
    plt.grid(True)
    
    # Ajustar espaçamento e salvar imagem
    plt.tight_layout()
    plt.savefig(f"out/quality_analysis_{axis_name}.png", dpi=150)
    plt.close()

def main():
    print("Iniciando extração de Fase com Desembrulho Temporal (Gray Code)...")
    
    dir_gray = os.path.expanduser("~/Desktop/sim_deflectometria/PMD/Gray_Code")
    dir_phase = os.path.expanduser("~/Desktop/sim_deflectometria/PMD/Phase")
    
    os.makedirs("out", exist_ok=True)
    
    dummy_cam_matrix = np.eye(3)
    dummy_dist = np.zeros(5)
    pmd = PMDProcessing(camera_matrix=dummy_cam_matrix, dist_coeffs=dummy_dist)

    print(f"Processando Fase a partir de: {dir_phase} ...")
    imgs_phase = load_images_from_directory(dir_phase, prefix="cam0_phase_step")
    
    print(f"Processando Gray Code a partir de: {dir_gray} ...")
    imgs_gray = load_images_from_directory(dir_gray, prefix="cam0_gray_bit")
    
    if len(imgs_phase) > 0 and len(imgs_gray) > 0:
        wrapped, mod, bg_phase = pmd.decode_nstep_phase(imgs_phase)
        np.save("out/wrapped_phase.npy", wrapped)
        cv2.imwrite("out/wrapped_phase.png", cv2.normalize(wrapped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        print("-> Fase salva em 'out/wrapped_phase.npy'")
        
        # Carregar imagem branca (se existir) para definir o limiar de binarização do Gray Code
        white_path = os.path.join(dir_gray, "cam0_gray_white.png")
        if os.path.exists(white_path):
            img_white = cv2.imread(white_path, cv2.IMREAD_GRAYSCALE)
            threshold_bg = img_white.astype(np.float32) * 0.5
            print("-> Usando 'cam0_gray_white.png' para o cálculo do limiar (threshold = 0.5 * white).")
        else:
            threshold_bg = bg_phase
            print("-> Imagem branca não encontrada, usando a média das franjas senoidais como limiar.")
        
        # Desembrulho Temporal (Gray Code)
        print("-> Decodificando sequencia Gray Code...")
        k = pmd.decode_graycode(imgs_gray, background=threshold_bg)
        np.save("out/fringe_order_k.npy", k)
        cv2.imwrite("out/fringe_order_k.png", cv2.normalize(k, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        
        print("-> Realizando desembrulho temporal...")
        unwrapped = pmd.graycode_unwrapping(wrapped, k, use_phase_guidance=True)
        
        # Aplicar mascara de qualidade e ruído (SNR baixo)
        valid_mod = mod[mod > 1e-3]
        if len(valid_mod) > 0:
            mod_threshold = np.percentile(valid_mod, 95) * 0.1
        else:
            mod_threshold = 1.0
        mask = mod > mod_threshold
        unwrapped = np.where(mask, unwrapped, np.nan)
        
        np.save("out/unwrapped_phase.npy", unwrapped)
        print("-> Fase Absoluta Desembrulhada salva em 'out/unwrapped_phase.npy'")
        
        evaluate_quality(wrapped, mod, 'X', unwrapped)
        print("-> Análise de qualidade salva em 'out/quality_analysis_X.png'")
    else:
        print(f"-> Imagens não encontradas. Verifique os diretórios e prefixos.")
        
    print("\nProcessamento concluído.")

if __name__ == "__main__":
    main()
