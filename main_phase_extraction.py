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
    print("Iniciando extração de Fase para Deflectometria (Single Frequency)...")
    
    dir_x = os.path.expanduser("~/Desktop/sim_deflectometria/vertical")
    dir_y = os.path.expanduser("~/Desktop/sim_deflectometria/horizontal")
    
    os.makedirs("out", exist_ok=True)
    
    dummy_cam_matrix = np.eye(3)
    dummy_dist = np.zeros(5)
    pmd = PMDProcessing(camera_matrix=dummy_cam_matrix, dist_coeffs=dummy_dist)

    # ----- EIXO X -----
    print(f"Processando Eixo X a partir de: {dir_x} ...")
    imgs_x = load_images_from_directory(dir_x)
    
    if len(imgs_x) > 0:
        wrapped_x, mod_x, _ = pmd.decode_nstep_phase(imgs_x)
        np.save("out/wrapped_phase_x.npy", wrapped_x)
        cv2.imwrite("out/wrapped_phase_x.png", cv2.normalize(wrapped_x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        print("-> Fase do Eixo X salva em 'out/wrapped_phase_x.npy'")
        
        # Desembrulho Espacial (Spatial Unwrapping)
        print("-> Realizando desembrulho espacial 2D (DCT)...")
        unwrapped_x = pmd.spatial_unwrapping(wrapped_x, mod_x)
        np.save("out/unwrapped_phase_x.npy", unwrapped_x)
        print("-> Fase Absoluta Desembrulhada salva em 'out/unwrapped_phase_x.npy'")
        
        evaluate_quality(wrapped_x, mod_x, 'X', unwrapped_x)
        print("-> Análise de qualidade do Eixo X salva em 'out/quality_analysis_X.png'")
    else:
        print(f"-> Imagens do Eixo X não encontradas ou diretório incorreto.")

    # ----- EIXO Y -----
    print(f"\nProcessando Eixo Y a partir de: {dir_y} ...")
    imgs_y = load_images_from_directory(dir_y)
    
    if len(imgs_y) > 0:
        wrapped_y, mod_y, _ = pmd.decode_nstep_phase(imgs_y)
        np.save("out/wrapped_phase_y.npy", wrapped_y)
        cv2.imwrite("out/wrapped_phase_y.png", cv2.normalize(wrapped_y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        print("-> Fase do Eixo Y salva em 'out/wrapped_phase_y.npy'")
        
        # Desembrulho Espacial (Spatial Unwrapping)
        print("-> Realizando desembrulho espacial 2D (DCT)...")
        unwrapped_y = pmd.spatial_unwrapping(wrapped_y, mod_y)
        np.save("out/unwrapped_phase_y.npy", unwrapped_y)
        print("-> Fase Absoluta Desembrulhada salva em 'out/unwrapped_phase_y.npy'")
        
        evaluate_quality(wrapped_y, mod_y, 'Y', unwrapped_y)
        print("-> Análise de qualidade do Eixo Y salva em 'out/quality_analysis_Y.png'")
    else:
        print(f"-> Imagens do Eixo Y não encontradas ou diretório incorreto.")
        
    print("\nProcessamento concluído.")

if __name__ == "__main__":
    main()
