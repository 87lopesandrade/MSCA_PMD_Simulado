# Guia de Utilização - MSCA PMD Simulado

Este repositório contém uma versão simplificada do módulo de Deflectometria por Medição de Fase (PMD) da arquitetura MSCA, voltada especificamente para **simulação**. O objetivo principal é receber imagens renderizadas de padrões de franjas (ex: via Blender) e processá-las para extrair a **fase** nos eixos X e Y.

Diferente do projeto completo, este repositório **não exige calibração estéreo real ou acesso ao hardware da Jetson Nano**. Ele foi projetado para rodar os algoritmos de _Phase Shifting_ puramente no ambiente virtual.

---

## 1. Configuração do Ambiente

Instale as dependências necessárias. Este projeto foi desenhado para rodar utilizando apenas a CPU, sendo totalmente compatível com Windows, Linux e Mac (incluindo Apple Silicon):

```bash
pip install -r requirements.txt
```

---

## 2. Preparação das Imagens de Simulação

O script principal espera imagens simuladas de padrões de franjas defasadas (N-step) utilizando **frequência única** (Single Frequency). 

O script já está configurado para ler as imagens (formato `.png`) diretamente das suas pastas no Desktop:

- **Eixo X (Franjas Verticais)**: `~/Desktop/sim_deflectometria/vertical/`
- **Eixo Y (Franjas Horizontais)**: `~/Desktop/sim_deflectometria/horizontal/`

*Certifique-se de que cada pasta contenha o número de imagens correspondente à quantidade de deslocamentos de fase (ex: 4, 8 ou 16 passos) e que não existam outras imagens `.png` perdidas nestas pastas.*

---

## 3. Passo a Passo de Execução

### Passo 1: Execução
No terminal, execute o script principal de extração de fase:

```bash
python main_phase_extraction.py
```

### Passo 2: Verificação de Resultados
Ao finalizar a execução, o script criará uma pasta `out/` e salvará os seguintes arquivos:

- `wrapped_phase_x.npy` e `wrapped_phase_y.npy`: Matrizes brutas com os valores da fase matemática "embrulhada" (variando de -π a π). 
- `wrapped_phase_x.png` e `wrapped_phase_y.png`: Imagens renderizadas da fase normalizadas (0-255) para que você consiga visualizá-las graficamente.

**Nota importante sobre Fase Absoluta:**
Como você está simulando com apenas uma frequência de franjas (sem usar heterodinação), o algoritmo extrai a fase "embrulhada" (wrapped phase). 
- Se a sua frequência de renderização for igual a **1** (ou seja, exatamente um período senoidal que preenche a tela inteira), essa fase resultante **já equivale à fase absoluta** do sistema e pode ser usada em cálculos de ray tracing.
- Se a sua frequência for maior que 1, você precisará aplicar um algoritmo de desembrulho espacial (spatial unwrapping) posteriormente, caso decida prosseguir com o pipeline 3D completo.
