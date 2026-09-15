import cv2
import numpy as np
from typing import Tuple, List
from scipy.fft import dctn, idctn
from numba import njit, prange

@njit(parallel=True)
def _numba_phase_guided_complementary(phi_wrap: np.ndarray, k_gray: np.ndarray) -> np.ndarray:
    """
    Filtro de monotonicidade otimizado em Numba para Gray Code Complementar.
    Transições ideais ocorrem em -pi/2 e pi/2.
    Zonas seguras são ao redor de -pi, 0 e pi.
    """
    H, W = phi_wrap.shape
    k_corrected = np.zeros_like(k_gray)
    pi = np.pi
    
    for i in prange(H):
        k_confiavel = k_gray[i, 0]
        
        # Inicializa k_confiavel com o primeiro pixel seguro
        for j in range(W):
            phi = phi_wrap[i, j]
            is_safe = (phi <= -3*pi/4) or (phi >= -pi/4 and phi <= pi/4) or (phi >= 3*pi/4)
            if is_safe:
                k_confiavel = k_gray[i, j]
                break
                
        for j in range(W):
            phi = phi_wrap[i, j]
            
            # Zonas Seguras (longe das transições -pi/2 e pi/2)
            is_safe = (phi <= -3*pi/4) or (phi >= -pi/4 and phi <= pi/4) or (phi >= 3*pi/4)
            
            if is_safe:
                k_confiavel = k_gray[i, j]
                k_corr = k_confiavel
            else:
                # Zonas de Risco (perto das transições)
                if phi > -3*pi/4 and phi < -pi/4:
                    # Transição em -pi/2
                    if phi < -pi/2.0:
                        k_corr = k_confiavel
                    else:
                        k_corr = k_confiavel + 1
                elif phi > pi/4 and phi < 3*pi/4:
                    # Transição em pi/2
                    if phi < pi/2.0:
                        k_corr = k_confiavel
                    else:
                        k_corr = k_confiavel + 1
                else:
                    k_corr = k_confiavel
                    
            k_corrected[i, j] = k_corr
            
    return k_corrected

class PMDProcessing:
    """
    Módulo para processamento das imagens de Deflectometria PMD.
    Extrai fase via N-step, realiza unwrapping temporal (Heterodyne)
    e calcula vetores normais precisos baseados na nuvem de pontos estéreo preexistente.
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Inicializa o processador.
        
        Args:
            camera_matrix: Matriz intrínseca 3x3 da câmera.
            dist_coeffs: Coeficientes de distorção.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def decode_nstep_phase(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decodifica a fase embrulhada (wrapped) a partir de N imagens deslocadas.
        
        Args:
            images: Lista de N imagens em tons de cinza ou float.
            
        Returns:
            Tuple (fase_embrulhada, modulacao, intensidade_fundo).
            - fase_embrulhada: Fase entre -pi e pi.
            - modulacao: Amplitude da franja (útil para máscara de SNR).
            - intensidade_fundo: Intensidade média.
        """
        N = len(images)
        img_stack = np.array(images, dtype=np.float32)
        
        # Cria matrizes para os somatórios
        sum_sin = np.zeros_like(images[0], dtype=np.float32)
        sum_cos = np.zeros_like(images[0], dtype=np.float32)
        sum_I = np.zeros_like(images[0], dtype=np.float32)
        
        for i in range(N):
            phase_shift = (2.0 * np.pi * i) / N
            sum_sin += img_stack[i] * np.sin(phase_shift)
            sum_cos += img_stack[i] * np.cos(phase_shift)
            sum_I += img_stack[i]
            
        wrapped_phase = np.arctan2(-sum_sin, sum_cos)
        modulation = (2.0 / N) * np.sqrt(sum_sin**2 + sum_cos**2)
        background = sum_I / N
        
        return wrapped_phase, modulation, background

    def heterodyne_unwrapping(self, phase1: np.ndarray, freq1: float, phase2: np.ndarray, freq2: float) -> np.ndarray:
        """
        Realiza o desembrulho temporal (Heterodyne Unwrapping) usando duas frequências.
        Frequência 1 (freq1) deve ser maior (período menor, alta resolução).
        Frequência 2 (freq2) deve ser menor (período maior) de modo que a diferença cubra a tela.
        
        Args:
            phase1: Fase embrulhada de alta frequência (-pi a pi).
            freq1: Número de períodos da phase1 na tela.
            phase2: Fase embrulhada de baixa frequência (-pi a pi).
            freq2: Número de períodos da phase2 na tela.
            
        Returns:
            Fase absoluta contínua (unwrapped).
        """
        # Fase da frequência batimento (beat frequency)
        beat_phase = phase1 - phase2
        beat_phase = np.arctan2(np.sin(beat_phase), np.cos(beat_phase)) # Re-wrap
        
        # Frequência de batimento
        beat_freq = freq1 - freq2
        
        # Estimar a ordem de franja (k) para a fase de alta frequência
        # phase_abs = phase1 + 2*pi*k
        # k = round( (beat_phase * (freq1 / beat_freq) - phase1) / (2*pi) )
        ratio = freq1 / beat_freq if beat_freq != 0 else 1.0
        
        k = np.round((beat_phase * ratio - phase1) / (2.0 * np.pi))
        
        absolute_phase = phase1 + 2.0 * np.pi * k
        return absolute_phase

    def compute_normals_from_phase(self, absolute_phase_x: np.ndarray, absolute_phase_y: np.ndarray, 
                                   period_x_mm: float, period_y_mm: float, 
                                   monitor_rvec: np.ndarray, monitor_tvec: np.ndarray,
                                   points_3d: np.ndarray) -> np.ndarray:
        """
        Calcula as normais da superfície para cada pixel a partir das fases absolutas 
        e da nuvem de pontos 3D preexistente (via projeção de franjas estéreo).
        
        Args:
            absolute_phase_x: Fase absoluta em X.
            absolute_phase_y: Fase absoluta em Y.
            period_x_mm: Período da franja absoluta em X convertido para milímetros na tela.
            period_y_mm: Período da franja absoluta em Y convertido para milímetros na tela.
            monitor_rvec: Rotação do monitor no sistema da câmera.
            monitor_tvec: Translação do monitor no sistema da câmera.
            points_3d: Matriz de pontos 3D da superfície (H, W, 3) calculados via estéreo.
            
        Returns:
            Matriz de normais (H, W, 3) associada a cada ponto 3D.
        """
        H, W = absolute_phase_x.shape
        normals = np.zeros((H, W, 3), dtype=np.float32)
        
        # Coordenadas físicas na tela a partir da fase
        x_lcd = (absolute_phase_x / (2.0 * np.pi)) * period_x_mm
        y_lcd = (absolute_phase_y / (2.0 * np.pi)) * period_y_mm
        
        R_mon, _ = cv2.Rodrigues(monitor_rvec)
        monitor_tvec_flat = monitor_tvec.flatten()
        
        # Iterar sobre a matriz para calcular a bissetriz
        # (Em produção de alta performance, isto pode ser completamente vetorizado no NumPy)
        for v in range(H):
            for u in range(W):
                p_surf = points_3d[v, u]
                
                # Verifica se o ponto 3D é válido
                if np.isnan(p_surf).any() or (p_surf[0] == 0 and p_surf[1] == 0 and p_surf[2] == 0):
                    continue
                    
                # Vetor da superfície para a câmera (que está na origem)
                v_cam = -p_surf
                norm_cam = np.linalg.norm(v_cam)
                if norm_cam > 1e-6:
                    v_cam = v_cam / norm_cam
                else:
                    continue
                    
                # Coordenada 3D da tela no sistema da câmera
                p_lcd_local = np.array([x_lcd[v, u], y_lcd[v, u], 0.0], dtype=np.float32)
                p_lcd_cam = monitor_tvec_flat + R_mon @ p_lcd_local
                
                # Vetor da superfície para a tela
                v_lcd = p_lcd_cam - p_surf
                norm_lcd = np.linalg.norm(v_lcd)
                if norm_lcd > 1e-6:
                    v_lcd = v_lcd / norm_lcd
                else:
                    continue
                
                # Normal = bissetriz de v_cam e v_lcd
                n = v_cam + v_lcd
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-6:
                    n = n / n_norm
                
                # A normal deve apontar em direção à câmera
                if n[2] > 0:
                    n = -n
                    
                normals[v, u] = n
                
        return normals

    def spatial_unwrapping(self, wrapped_phase: np.ndarray, modulation: np.ndarray, mod_threshold: float = None) -> np.ndarray:
        """
        Realiza o desembrulho espacial 2D da fase utilizando Mínimos Quadrados via Transformada Discreta de Cosseno (DCT).
        Aplica uma máscara de confiabilidade baseada na modulação para remover o ruído de fundo.
        
        Args:
            wrapped_phase: Mapa de fase embrulhada (2D, -pi a pi).
            modulation: Mapa de modulação/SNR (2D).
            mod_threshold: Limiar de modulação. Se None, é calculado usando percentil 95.
            
        Returns:
            Fase contínua (unwrapped) onde os pixels de ruído recebem np.nan.
        """
        H, W = wrapped_phase.shape
        
        # 1. Gradientes da Fase Embrulhada
        def wrap_diff(diff):
            return diff - 2.0 * np.pi * np.round(diff / (2.0 * np.pi))
            
        dx = np.zeros((H, W), dtype=np.float32)
        dy = np.zeros((H, W), dtype=np.float32)
        
        dx[:, :-1] = wrap_diff(wrapped_phase[:, 1:] - wrapped_phase[:, :-1])
        dy[:-1, :] = wrap_diff(wrapped_phase[1:, :] - wrapped_phase[:-1, :])
        
        # 2. Laplaciano (Divergente dos Gradientes)
        rho = np.zeros((H, W), dtype=np.float32)
        rho[:, 1:] += dx[:, :-1]
        rho[:, :-1] -= dx[:, :-1]
        rho[1:, :] += dy[:-1, :]
        rho[:-1, :] -= dy[:-1, :]
        
        # 3. Transformada Discreta de Cosseno 2D
        dct_rho = dctn(rho, type=2, norm='ortho')
        
        # 4. Solução da Equação de Poisson no Domínio da Frequência
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
        
        denom = 2.0 * np.cos(np.pi * x_grid / W) + 2.0 * np.cos(np.pi * y_grid / H) - 4.0
        # Evitar divisão por zero na frequência zero (u=0, v=0)
        denom[0, 0] = 1.0
        
        dct_phi = dct_rho / denom
        dct_phi[0, 0] = 0.0  # Remove a componente contínua
        
        # 5. Transformada Inversa
        unwrapped_phase = idctn(dct_phi, type=2, norm='ortho')
        
        # 6. Mascaramento baseado na Modulação
        if mod_threshold is None:
            # Calcula threshold dinâmico ignorando o fundo exato
            valid_mod = modulation[modulation > 1e-3]
            if len(valid_mod) > 0:
                # Usa 10% do percentil 95 para não ser rigoroso demais
                mod_threshold = np.percentile(valid_mod, 95) * 0.1
            else:
                mod_threshold = 1.0
                
        # Mascara onde a modulação for menor que o limiar (fundo/ruído)
        mask = modulation > mod_threshold
        unwrapped_phase = np.where(mask, unwrapped_phase, np.nan)
        
        return unwrapped_phase

    def decode_graycode(self, gray_images: List[np.ndarray], background: np.ndarray = None) -> np.ndarray:
        """
        Decodifica a sequência de código Gray para encontrar a ordem da franja (QSI/k).
        
        Args:
            gray_images: Lista de imagens do código Gray (ordenadas do bit mais significativo para o menos).
            background: Imagem de intensidade de fundo (threshold) para binarização.
            
        Returns:
            Matriz de inteiros representando a ordem da franja (k).
        """
        if background is None:
            # Se não houver threshold, tenta usar a média das imagens Gray
            img_stack = np.array(gray_images, dtype=np.float32)
            background = np.mean(img_stack, axis=0)
            
        # 1. Binarização usando o threshold
        binary_images = [(img > background).astype(np.uint8) for img in gray_images]
        
        # 2. Conversão de Gray para Binário normal
        binary_normal = []
        binary_normal.append(binary_images[0])
        for i in range(1, len(binary_images)):
            binary_normal.append(np.logical_xor(binary_normal[i-1], binary_images[i]).astype(np.uint8))
            
        # 3. Conversão de Binário para Inteiro (QSI / k)
        k = np.zeros_like(gray_images[0], dtype=np.int32)
        num_bits = len(binary_images)
        for i, b_img in enumerate(binary_normal):
            # bit 0 é o MSB
            weight = 1 << (num_bits - 1 - i)
            k += b_img.astype(np.int32) * weight
            
        return k

    def graycode_unwrapping(self, wrapped_phase: np.ndarray, k: np.ndarray, use_phase_guidance: bool = True) -> np.ndarray:
        """
        Desembrulha a fase usando a ordem de franja k extraída do código Gray.
        Assume Padrão Gray Code Complementar (Transição dupla por período de franja).
        
        Args:
            wrapped_phase: Fase embrulhada entre -pi e pi.
            k: Ordem da franja QSI (transita a cada pi/2).
            use_phase_guidance: Utiliza o filtro guiado para corrigir o atraso do k causado por desfoque.
            
        Returns:
            Fase absoluta contínua.
        """
        if use_phase_guidance:
            k_corrected = _numba_phase_guided_complementary(wrapped_phase, k)
            k_float = k_corrected.astype(np.float32)
        else:
            k_float = k.astype(np.float32)
            
        absolute_phase = np.zeros_like(wrapped_phase)
        
        # O código gerado (baseado no Voris) possui duas regiões de Gray Code para cada franja.
        # Ou seja, o valor 'k' (QSI) incrementa duas vezes por período da senóide.
        # As quebras ideais do k ocorrem em -pi/2 e +pi/2, evitando a borda ruidosa de +-pi.
        
        mask_left1 = wrapped_phase <= -np.pi / 2.0
        mask_left2 = (wrapped_phase > -np.pi / 2.0) & (wrapped_phase < np.pi / 2.0)
        mask_left3 = wrapped_phase >= np.pi / 2.0
        
        # Desembrulho robusto de fase (Spatial-Temporal shift):
        absolute_phase[mask_left1] = wrapped_phase[mask_left1] + 2.0 * np.pi * np.floor((k_float[mask_left1] + 1.0) / 2.0)
        absolute_phase[mask_left2] = wrapped_phase[mask_left2] + 2.0 * np.pi * np.floor(k_float[mask_left2] / 2.0)
        absolute_phase[mask_left3] = wrapped_phase[mask_left3] + 2.0 * np.pi * (np.floor((k_float[mask_left3] + 1.0) / 2.0) - 1.0)
        
        return absolute_phase

    def graycode_unwrapping_classic(self, wrapped_phase: np.ndarray, k: np.ndarray, use_phase_guidance: bool = True) -> np.ndarray:
        """
        Desembrulha a fase para o método Gray Code Clássico (relação 1:1, transição em +-pi).
        
        Args:
            wrapped_phase: Fase embrulhada entre -pi e pi.
            k: Matriz contendo a ordem das franjas (QSI).
            use_phase_guidance: Liga/Desliga o filtro de correção guiado pela fase (Phase-Guided Correction).
                                Otimizado via Numba para alta performance em imagens de alta resolução.
                                
        Returns:
            Matriz de Fase Desembrulhada contínua.
        """
        if use_phase_guidance:
            # Chama a função otimizada em Numba
            return _numba_phase_guided_unwrapping(wrapped_phase, k)
        else:
            # Equação clássica matemática direta (susceptível a spikes nas bordas devido a desfoque)
            return wrapped_phase + (k.astype(np.float32) * 2.0 * np.pi)


