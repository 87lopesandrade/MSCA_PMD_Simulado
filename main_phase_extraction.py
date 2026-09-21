import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_gc_order_v(width, px_f, n_bits):
    """
    Gera a ordem real do Gray Code reproduzindo a mesma técnica do VORIS.
    No VORIS, isso é feito desenhando a imagem de Gray Code ideal e extraindo 
    seus valores únicos.
    """
    width_list = [element for element in np.arange(2 ** n_bits, dtype=np.uint8) for _ in range(int(px_f / 2))]
    graycode_list = [n ^ (n >> 1) for n in width_list]
    
    # Get unique values while preserving order
    _, indices = np.unique(graycode_list, return_index=True)
    sorted_indices = np.argsort(indices)
    sorted_qsi_val = np.array(graycode_list)[indices][sorted_indices]
    return sorted_qsi_val

def remap_qsi_image(qsi_image, real_qsi_order):
    """
    Remapeia os valores QSI lidos (baseados no raw gray code binário) para
    índices ordinais (0, 1, 2, ...). Equivale à conversão Gray -> Binary.
    """
    max_order_val = int(np.max(real_qsi_order))
    max_qsi_val = int(np.max(qsi_image))
    max_val = max(max_order_val, max_qsi_val)
    
    lut = np.zeros(max_val + 1, dtype=np.int64)
    lut[real_qsi_order] = np.arange(len(real_qsi_order), dtype=np.int64)
    
    valid_mask = (qsi_image >= 0) & (qsi_image <= max_val)
    safe_qsi = np.where(valid_mask, qsi_image, 0)
    remapped_qsi_image = np.where(valid_mask, lut[safe_qsi], 0)
    
    return remapped_qsi_image

def process_camera(cam_dir, side):
    print(f"--- Processando Câmera {side} ---")
    # Read L000 to L015
    imgs = []
    prefix = 'L' if side == 'left' else 'R'
    for i in range(16):
        path = os.path.join(cam_dir, f"{prefix}{i:03d}.png")
        if not os.path.exists(path):
            print(f"Erro: não achou {path}")
            return None
        imgs.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32))
        
    gc_imgs = np.stack(imgs[:8], axis=-1)   # 8 canais (L000 a L007)
    ph_imgs = np.stack(imgs[8:], axis=-1)   # 8 canais (L008 a L015)
    
    # 1. Calcular Phi e Modulação
    num_channels = ph_imgs.shape[-1]
    indices = np.arange(1, num_channels + 1, dtype=np.float32)
    angle = 2.0 * np.pi * indices / float(num_channels)
    
    sin_values = np.sin(angle)
    cos_values = np.cos(angle)
    
    sin_contributions = np.sum(ph_imgs * sin_values, axis=2)
    cos_contributions = np.sum(ph_imgs * cos_values, axis=2)
    
    phi_image = np.arctan2(-sin_contributions, cos_contributions)
    modulation_map = np.sqrt(sin_contributions**2 + cos_contributions**2) / num_channels
    
    # 2. Calcular QSI bruto (Raw Binary from Gray Code)
    white_value = gc_imgs[:, :, 0]
    white_value = np.clip(white_value, 1e-6, None)
    
    bit_values = gc_imgs[:, :, 2:] / white_value[..., np.newaxis]
    bit_values = (bit_values > 0.5).astype(np.int64)
    
    num_bits = bit_values.shape[-1] # Deve ser 6 (L002 a L007)
    powers = 2 ** np.arange(num_bits - 1, -1, -1, dtype=np.int64)
    qsi_image = np.sum(bit_values * powers, axis=-1)
    
    # 3. Remapear QSI para índice real
    real_qsi_order = get_gc_order_v(width=2448, px_f=64, n_bits=6)
    remaped_qsi_image = remap_qsi_image(qsi_image, real_qsi_order)
    
    # 4. Desembrulho de Fase (Tiago Loureiro, idêntico ao VORIS)
    remap_float = remaped_qsi_image.astype(np.float32)
    abs_phi_image = np.zeros_like(phi_image)
    
    m1 = phi_image <= -np.pi / 2.0
    m2 = (phi_image > -np.pi / 2.0) & (phi_image < np.pi / 2.0)
    m3 = phi_image >= np.pi / 2.0
    
    abs_phi_image[m1] = phi_image[m1] + 2.0 * np.pi * np.floor((remap_float[m1] + 1.0) / 2.0) + np.pi
    abs_phi_image[m2] = phi_image[m2] + 2.0 * np.pi * np.floor(remap_float[m2] / 2.0) + np.pi
    abs_phi_image[m3] = phi_image[m3] + 2.0 * np.pi * (np.floor((remap_float[m3] + 1.0) / 2.0) - 1.0) + np.pi
    
    return abs_phi_image, phi_image, modulation_map, remaped_qsi_image

def main():
    data_dir = os.path.expanduser("~/Desktop/sim_deflectometria/PMD_Voris/out")
    out_dir = os.path.join(data_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    for side in ['left', 'right']:
        cam_dir = os.path.join(data_dir, side)
        if not os.path.exists(cam_dir):
            continue
            
        res = process_camera(cam_dir, side)
        if res is None:
            continue
            
        abs_phi, phi, mod, qsi = res
        
        # Mascara do espelho
        # Em VORIS, costumam usar a modulação. Como no código antigo a máscara era mod > 5.0, 
        # mas lá mod era dividido por 4.0 em vez de 8.0, vamos usar mod > 1.0
        mask = mod > 1.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
        
        abs_phi_masked = np.where(mask_eroded, abs_phi, np.nan)
        
        # Save NPY
        np.save(os.path.join(out_dir, f"abs_phi_{side}.npy"), abs_phi_masked)
        
        # Plot
        plt.figure(figsize=(15, 12))
        
        # 1D Phase vs QSI Remap
        row = abs_phi.shape[0] // 2
        
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(abs_phi_masked[row, :], color='red', label='Abs Phi')
        ax1.set_ylabel('Abs Phi Image (rad)', color='red')
        ax1.set_title(f'Abs Phi Image {side} 1D (Row {row})')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(qsi[row, :], color='blue', alpha=0.5, label='Remapped QSI')
        ax2.set_ylabel('Remapped QSI Image', color='blue')
        
        # 2D Abs Phi
        plt.subplot(3, 2, 3)
        plt.imshow(abs_phi_masked, cmap='gray')
        plt.colorbar()
        plt.title(f'Abs Phi Image {side} 2D')
        
        # 2D Phi Wrapped
        plt.subplot(3, 2, 4)
        plt.imshow(np.where(mask_eroded, phi, np.nan), cmap='gray')
        plt.colorbar()
        plt.title(f'Wrapped Phi Image {side} 2D')
        
        # 2D Modulation Map
        plt.subplot(3, 2, 5)
        plt.imshow(mod, cmap='jet')
        plt.colorbar()
        plt.title(f'Modulation Map {side}')
        
        # 2D QSI Map
        plt.subplot(3, 2, 6)
        plt.imshow(np.where(mask_eroded, qsi, np.nan), cmap='viridis')
        plt.colorbar()
        plt.title(f'Remapped QSI Image {side} 2D')
        
        plt.tight_layout()
        plt_path = os.path.join(out_dir, f"analysis_{side}.png")
        plt.savefig(plt_path, dpi=150)
        plt.close()
        print(f"Salvo: {plt_path}")

if __name__ == '__main__':
    main()
