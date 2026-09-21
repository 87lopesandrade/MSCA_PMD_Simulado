# Guia de Utilização - MSCA PMD Simulado (Phase Shifting + Gray Code)

Este repositório contém o módulo de Deflectometria por Medição de Fase (PMD) focado no processamento de imagens simuladas (ex: via Blender). O projeto implementa as técnicas de desembrulho espacial de fase combinando **Phase Shifting** e **Gray Code**, adotando a matemática exata do projeto VORIS.

Diferente do projeto completo, este repositório é simplificado, rodando apenas na CPU sem exigir hardware ou câmeras externas.

---

## 1. Configuração do Ambiente

O projeto é mantido através de um ambiente virtual em Python. Certifique-se de que os pacotes necessários estejam instalados (você pode utilizar o arquivo de dependências se precisar reinstalar):

```bash
pip install -r requirements.txt
```

As principais bibliotecas são `numpy`, `opencv-python` e `matplotlib`.

---

## 2. Preparação das Imagens

O script `main_phase_extraction.py` espera que as imagens adquiridas/simuladas existam nos seguintes diretórios:

- **Câmera Esquerda:** `~/Desktop/sim_deflectometria/PMD_Voris/out/left/`
- **Câmera Direita:** `~/Desktop/sim_deflectometria/PMD_Voris/out/right/`

### Estrutura de Nomenclatura das Imagens
Para a Câmera Esquerda (prefixo `L`) e Direita (prefixo `R`), deve-se ter as imagens nomeadas numericamente com formato PNG, totalizando 16 imagens por câmera:

- **Imagens 0 a 7 (Gray Code)**:
  - `L000.png` / `R000.png`: Imagem branca para normalização e binarização.
  - `L001.png` / `R001.png`: Imagem totalmente preta (geralmente ignorada na binarização direta, mas presente na sequência).
  - `L002.png` a `L007.png` (ou R): Os 6 bits da sequência de Gray Code, projetados do menos significativo ao mais significativo.

- **Imagens 8 a 15 (Phase Shifting)**:
  - `L008.png` a `L015.png` (ou R): Padrão senoidal de fase deslocada (N=8 passos).

---

## 3. Passo a Passo de Execução

No terminal, estando na raiz deste projeto, execute o script de extração:

```bash
python main_phase_extraction.py
```

### 4. Verificação de Resultados

Ao finalizar o processamento, os resultados serão salvos no diretório:
`~/Desktop/sim_deflectometria/PMD_Voris/out/results/`

Lá você encontrará os arquivos:

1. **`abs_phi_left.npy` e `abs_phi_right.npy`**: 
   Matrizes brutas (em formato NumPy) contendo os valores da fase matemática "desembrulhada" e mapeada (Fase Absoluta). Os ruídos de fundo e regiões onde a modulação da franja não chegou foram filtrados (marcados como `NaN`). Essas matrizes são adequadas para continuar o pipeline de reconstrução 3D ou calibração de malha.

2. **`analysis_left.png` e `analysis_right.png`**:
   Imagens de verificação (plots) que incluem:
   - Um perfil 1D (perfil transversal no centro da imagem) mostrando o mapa contínuo da fase absoluta versus o Gray Code.
   - Um mapa em 2D da fase envolta (Wrapped Phi).
   - Um mapa da fase absoluta espacial, mostrando que os pulos da onda senoidal de 2π foram totalmente resolvidos de maneira suave e correta, baseando-se no Gray Code (Tiago Loureiro's spatial unwrapping).
   - Mapas auxiliares de Modulação e QSI Remapeado.
