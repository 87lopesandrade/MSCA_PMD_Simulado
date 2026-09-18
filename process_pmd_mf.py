import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/Users/igorlopesdeandrade/Desktop/sim_deflectometria/PMD_MF"
out_dir = os.path.join(data_dir, "out")
os.makedirs(out_dir, exist_ok=True)

frequencies = [1, 4, 32]
num_steps = 4

def decode_phase(cam_prefix, f_val):
    imgs = []
    for s in range(num_steps):
        fname = os.path.join(data_dir, f"{cam_prefix}_mf_f{f_val}_step{s}.png")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Imagem ausente: {fname}")
        imgs.append(cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float64))
    
    I0, I1, I2, I3 = imgs
    # 4-passos: I3 - I1 = 2*B*sin, I0 - I2 = 2*B*cos
    phi = np.arctan2(I3 - I1, I0 - I2)
    modulation = 0.5 * np.sqrt((I3 - I1)**2 + (I0 - I2)**2)
    return phi, modulation

def unwrap_hierarchical(cam_prefix):
    print(f"\n--- Processando Desembrulho Multi-Frequência: {cam_prefix} ---")
    
    # 1. Decodificar as 3 fases embrulhadas
    phases = {}
    modulations = {}
    for f in frequencies:
        p, m = decode_phase(cam_prefix, f)
        phases[f] = p
        modulations[f] = m
        print(f"-> Fase f={f} decodificada. Modulacao media: {np.mean(m):.1f}")
        
    # Mascara de modulação
    mod_high = modulations[32]
    mask = mod_high > 5.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    
    # 2. Desembrulho Hierarquico
    # CORREÇÃO CRÍTICA:
    # Como a franja unitária (f=1) varia de 0 a 2*pi ao longo do monitor,
    # o arctan2 normal corta em [-pi, pi), gerando um salto de -2*pi no centro da tela.
    # Usamos np.mod(phases[1], 2*pi) para obter a fase absoluta contínua de [0, 2*pi):
    Phi_abs = {1: np.mod(phases[1], 2.0 * np.pi)}
    
    # Desembrulhar hierarquicamente: f1 -> f2 (1 -> 4) e f2 -> f3 (4 -> 32)
    f_prev = 1
    for f_curr in [4, 32]:
        ratio = float(f_curr) / float(f_prev)
        phi_curr = phases[f_curr]
        Phi_prev = Phi_abs[f_prev]
        
        # Ordem k = round( (ratio * Phi_prev - phi_curr) / (2*pi) )
        k = np.round((ratio * Phi_prev - phi_curr) / (2.0 * np.pi))
        Phi_curr_abs = phi_curr + 2.0 * np.pi * k
        Phi_abs[f_curr] = Phi_curr_abs
        print(f"-> Transicao f={f_prev} -> f={f_curr} (Razao {ratio:.1f}x): Ordem k min={np.nanmin(k):.0f}, max={np.nanmax(k):.0f}")
        f_prev = f_curr
        
    Phi_final = Phi_abs[32]
    Phi_final[~mask_eroded] = np.nan
    
    # Salvar matriz de fase absoluta
    np.save(os.path.join(out_dir, f"unwrapped_phase_{cam_prefix}.npy"), Phi_final)
    
    # Avaliar linha central
    H, W = Phi_final.shape
    row = H // 2
    row_u = Phi_final[row, :]
    valid_x = np.where(~np.isnan(row_u))[0]
    
    if len(valid_x) > 0:
        x_sub = row_u[valid_x]
        diffs = np.diff(x_sub)
        # Verifica qualquer descontinuidade anormal (> pi)
        jumps = np.where(np.abs(diffs) > np.pi)[0]
        print(f"Linha {row}: Total pixels validos = {len(x_sub)} | Saltos anormais (> pi) = {len(jumps)}")
        if len(jumps) == 0:
            print("✓ SUCESSO TOTAL! FASE 100% CONTINUA, SEM NENHUM SALTO OU QUEDA!")
        else:
            print(f"Aviso: {len(jumps)} saltos encontrados em X: {valid_x[jumps]}")
            
    # Graficos
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(phases[32], cmap='jet')
    plt.title(f"Fase Embrulhada (f=32) - {cam_prefix}")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 3, 2)
    plt.imshow(Phi_final, cmap='viridis')
    plt.title(f"Fase Absoluta Continua (f=32) - {cam_prefix}")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.subplot(1, 3, 3)
    if len(valid_x) > 0:
        plt.plot(valid_x, x_sub, color='purple', linewidth=1.5)
        plt.title(f"Perfil da Linha {row} (100% Contínuo)")
        plt.xlabel("Pixel X")
        plt.ylabel("Fase Absoluta (rad)")
        plt.grid(True)
    plt.tight_layout()
    plot_file = os.path.join(out_dir, f"resultado_mf_{cam_prefix}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"-> Grafico salvo em: {plot_file}")

if __name__ == "__main__":
    unwrap_hierarchical("cam0")
    unwrap_hierarchical("cam1")
