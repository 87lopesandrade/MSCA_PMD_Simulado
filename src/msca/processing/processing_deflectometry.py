import cv2
import numpy as np
from typing import Tuple, List
from scipy.fft import dctn, idctn

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

