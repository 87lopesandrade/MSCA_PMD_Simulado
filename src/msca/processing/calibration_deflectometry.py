import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, List, Optional, Dict, Any

class PMDCalibration:
    """
    Classe para realizar a calibração geométrica de um sistema PMD (Phase Measuring Deflectometry).
    Utiliza um espelho de primeira superfície referenciado com marcadores ChArUco.
    Otimiza a pose do monitor minimizando o erro 2D no plano da tela.
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Inicializa o calibrador com os intrínsecos da câmera.
        
        Args:
            camera_matrix: Matriz intrínseca 3x3 da câmera.
            dist_coeffs: Coeficientes de distorção (k1, k2, p1, p2, k3).
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Parâmetros otimizados do monitor (rvec, tvec)
        self.monitor_rvec = np.zeros((3, 1))
        self.monitor_tvec = np.zeros((3, 1))

    def detect_charuco_pose(self, image: np.ndarray, charuco_dict: Any, charuco_board: Any) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Detecta um alvo ChArUco e estima a sua pose (rvec, tvec).
        Isto define o plano do espelho.
        
        Args:
            image: Imagem capturada pela câmera contendo o alvo ChArUco.
            charuco_dict: Dicionário Aruco utilizado (ex: cv2.aruco.DICT_4X4_50).
            charuco_board: Objeto cv2.aruco.CharucoBoard com a definição do tabuleiro.
            
        Returns:
            Tuple contendo: (sucesso_booleano, rvec, tvec).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # OpenCV > 4.6.0 syntax
        detector = cv2.aruco.CharucoDetector(charuco_board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if charuco_ids is not None and len(charuco_ids) > 3:
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, 
                self.camera_matrix, self.dist_coeffs, np.empty(1), np.empty(1)
            )
            if ret:
                return True, rvec, tvec
        
        return False, np.zeros((3,1)), np.zeros((3,1))

    def ray_tracing_reflection(self, u: float, v: float, mirror_rvec: np.ndarray, mirror_tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza o ray tracing de um pixel (u,v) da câmera refletindo no espelho.
        
        Args:
            u: Coordenada x na imagem (pixel).
            v: Coordenada y na imagem (pixel).
            mirror_rvec: Rotação do espelho.
            mirror_tvec: Translação do espelho.
            
        Returns:
            Tuple (ponto_intersecao_espelho, vetor_direcao_refletido) no sistema da câmera.
        """
        # Desfaz a distorção e converte para coordenadas normalizadas (plano z=1)
        # Nota: cv2.undistortPoints expects shape (N, 1, 2)
        pt_img = np.array([[[u, v]]], dtype=np.float32)
        pt_undistorted = cv2.undistortPoints(pt_img, self.camera_matrix, self.dist_coeffs)
        
        x_norm, y_norm = pt_undistorted[0, 0]
        
        # Vetor de direção visual a partir da origem da câmera (que está em 0,0,0)
        v_cam = np.array([x_norm, y_norm, 1.0])
        v_cam = v_cam / np.linalg.norm(v_cam)
        
        # Parâmetros do plano do espelho
        R_mirror, _ = cv2.Rodrigues(mirror_rvec)
        normal_mirror = R_mirror[:, 2] # O eixo Z do ChArUco é ortogonal ao plano
        point_mirror = mirror_tvec.flatten()
        
        # Interseção Raio-Plano: t = ( (P0 - L0) . n ) / ( l . n )
        # L0 = [0,0,0], l = v_cam, P0 = point_mirror, n = normal_mirror
        denom = np.dot(v_cam, normal_mirror)
        if abs(denom) < 1e-6:
            # Raio paralelo ao plano (teoricamente impossível na configuração física, mas por segurança)
            return np.zeros(3), np.zeros(3)
            
        t = np.dot(point_mirror, normal_mirror) / denom
        p_intersect = t * v_cam
        
        # Reflexão: r = d - 2*(d.n)*n
        # Mas para lei da reflexão, r e d devem apontar para fora do ponto de incidência.
        # v_cam aponta para a superfície.
        d = v_cam
        n = normal_mirror
        r = d - 2 * np.dot(d, n) * n
        r = r / np.linalg.norm(r)
        
        return p_intersect, r

    def intersect_ray_monitor(self, ray_origin: np.ndarray, ray_dir: np.ndarray, monitor_rvec: np.ndarray, monitor_tvec: np.ndarray) -> Optional[np.ndarray]:
        """
        Calcula a interseção de um raio 3D com o plano do monitor.
        Retorna a coordenada 2D local no plano do monitor.
        
        Args:
            ray_origin: Ponto 3D de origem do raio (ponto no espelho).
            ray_dir: Vetor 3D de direção do raio refletido.
            monitor_rvec: Rotação proposta para o monitor.
            monitor_tvec: Translação proposta para o monitor.
            
        Returns:
            Coordenada local 2D (x, y) no monitor (em mm), ou None se não interceptar.
        """
        R_mon, _ = cv2.Rodrigues(monitor_rvec)
        # Plano do monitor: Origem = tvec, Normal = Eixo Z do monitor
        normal_mon = R_mon[:, 2]
        p_mon = monitor_tvec.flatten()
        
        denom = np.dot(ray_dir, normal_mon)
        if abs(denom) < 1e-6:
            return None
            
        t = np.dot(p_mon - ray_origin, normal_mon) / denom
        
        if t < 0:
            return None # Monitor está atrás do espelho ao longo da direção do raio
            
        # Ponto de interseção no espaço 3D global da câmera
        p_intersect_3d = ray_origin + t * ray_dir
        
        # Converter para o sistema de coordenadas local do monitor
        # P_local = R_mon.T * (P_global - tvec)
        p_local_3d = R_mon.T @ (p_intersect_3d - p_mon)
        
        # Retorna apenas X e Y (Z deve ser aproximadamente 0 por definição do plano)
        return p_local_3d[0:2]

    def _cost_function(self, params: np.ndarray, points_img: np.ndarray, points_lcd_mm: np.ndarray, mirror_poses: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Função de custo para o otimizador Levenberg-Marquardt.
        Minimiza o erro 2D no plano do monitor.
        
        Args:
            params: [r1, r2, r3, tx, ty, tz] do monitor.
            points_img: Array Nx2 de coordenadas (u,v) na imagem da câmera (correspondentes aos pontos decodificados).
            points_lcd_mm: Array Nx2 de coordenadas reais (x,y) no LCD, em milímetros (obtidas da fase decodificada).
            mirror_poses: Lista de posições (rvec, tvec) do espelho para cada ponto, em paralelo a points_img.
            
        Returns:
            Resíduos 1D (achatado Nx2 -> 2N) da diferença entre as coordenadas preditas e medidas.
        """
        mon_rvec = params[0:3].reshape(3, 1)
        mon_tvec = params[3:6].reshape(3, 1)
        
        residuals = []
        for i in range(len(points_img)):
            u, v = points_img[i]
            x_lcd_measured, y_lcd_measured = points_lcd_mm[i]
            
            m_rvec, m_tvec = mirror_poses[i]
            
            # 1. Tracing Câmera -> Espelho
            p_mirror, r_dir = self.ray_tracing_reflection(u, v, m_rvec, m_tvec)
            
            # 2. Interseção Espelho -> Tela
            p_lcd_pred = self.intersect_ray_monitor(p_mirror, r_dir, mon_rvec, mon_tvec)
            
            if p_lcd_pred is not None:
                # 3. Erro no plano da tela 2D
                res_x = p_lcd_pred[0] - x_lcd_measured
                res_y = p_lcd_pred[1] - y_lcd_measured
            else:
                res_x, res_y = 1e6, 1e6 # Penalidade massiva se não intersecionar
                
            residuals.append(res_x)
            residuals.append(res_y)
            
        return np.array(residuals)

    def optimize_monitor_pose(self, points_img: np.ndarray, points_lcd_mm: np.ndarray, mirror_poses: List[Tuple[np.ndarray, np.ndarray]], initial_guess: np.ndarray = None) -> bool:
        """
        Executa a otimização não linear para calibrar a pose do monitor.
        
        Args:
            points_img: Nx2 pixels na imagem da câmera correspondentes a pixels da tela.
            points_lcd_mm: Nx2 coordenadas milimétricas no LCD, conhecidas via Phase Shifting absoluto.
            mirror_poses: Lista com N elementos (rvec, tvec) para a pose do espelho no instante da medição.
            initial_guess: [r1, r2, r3, tx, ty, tz]. Se None, assume zeros.
            
        Returns:
            True se a otimização convergir.
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
            
        res = least_squares(
            self._cost_function, 
            initial_guess, 
            args=(points_img, points_lcd_mm, mirror_poses),
            method='lm',
            verbose=2
        )
        
        if res.success:
            self.monitor_rvec = res.x[0:3].reshape(3, 1)
            self.monitor_tvec = res.x[3:6].reshape(3, 1)
            return True
            
        return False
