# PA1 — Segmentação de Instâncias com Aprendizado Profundo

## Integrantes

- Ana Júlia Amaro Pereira Rocha
- Maria Eduarda Mesquita Magalhães

---

## 1. Objetivo

Este projeto foi desenvolvido para a PA1 da disciplina de Aprendizado Profundo.

O objetivo é transformar uma arquitetura de segmentação semântica em um sistema capaz de realizar **segmentação de instâncias**, sem utilizar detectores baseados em propostas de regiões ou modelos prontos de instance segmentation.

A arquitetura principal utilizada foi uma **U-Net**, com uma representação baseada em:

- classificação de cada pixel em background, interior e boundary;
- geração de marcadores a partir da região de interior;
- utilização do algoritmo Watershed para separar objetos que se tocam.

O projeto também investiga limitações da abordagem por meio de ablações, inferência em mosaicos, análise de falhas e testes de robustez.

---

## 2. Dataset

Foi utilizado um dataset sintético de imagens de tamanho `128 × 128` contendo objetos elípticos com:

- diferentes tamanhos;
- diferentes posições;
- diferentes níveis de sobreposição/contato;
- ruído e variação de contraste.

Cada amostra possui:

- `image.png`: imagem de entrada;
- `semantic_mask.png`: máscara binária de foreground;
- `instance_mask.png`: máscara com identificadores individuais das instâncias.

O dataset utilizado nos experimentos contém 20 imagens e 243 instâncias.

Os objetos possuem tamanhos variados, com:

- área média: 268,30 pixels;
- área mediana: 236 pixels;
- largura máxima: 31 pixels;
- altura máxima: 31 pixels;
- diâmetro equivalente máximo: 29,10 pixels.

---

## 3. Arquitetura

A arquitetura principal é uma **U-Net**.

A U-Net possui quatro etapas de encoder, um bottleneck e quatro etapas de decoder com skip connections.

Para a segmentação semântica baseline, a rede produz um único canal correspondente à probabilidade de foreground.

Para a abordagem de segmentação de instâncias, a rede produz três classes:

```text
0 — background
1 — interior
2 — boundary
```

---

## Estrutura do código


```text
src/segmentation/
├── data/             # Dataset e pré-processamento
├── inference/        # Inferência em mosaicos e análise de campo receptivo
├── losses/           # Funções de perda
├── metrics/          # Métricas semânticas e de instâncias
├── models/           # U-Net e SegNet
└── postprocessing/   # Componentes conexos, fusão e Watershed
scripts/              # Treino, avaliação, visualizações e geração de dados
data/                 # Dados de entrada
checkpoints/          # Pesos treinados
output_dir/           # Resultados gerados
```

Execute os pontos de entrada a partir da raiz usando módulos Python. Por exemplo:

```bash
python -m scripts.train
python -m scripts.evaluate
python -m scripts.generate_synthetic_dataset
```

## Guia de arquivos Python

Os arquivos `__init__.py` em `src/`, `src/segmentation/` e em cada subpasta apenas
marcam os diretórios como pacotes Python; não contêm lógica de negócio. O mesmo vale
para `scripts/__init__.py`, que permite executar os pontos de entrada com
`python -m scripts.<nome>`.

### Núcleo reutilizável — `src/segmentation/`

| Arquivo | Responsabilidade |
| --- | --- |
| `data/dataset.py` | Define `SyntheticSegmentationDataset`, que lê `image.png`, `semantic_mask.png` e `instance_mask.png`, converte-os para tensores e disponibiliza cada amostra ao `DataLoader`. Também produz a máscara de borda quando solicitada. |
| `data/preprocessing.py` | Normaliza as imagens de entrada e cria a máscara com as três classes: background, interior e boundary. |
| `models/unet.py` | Implementa a U-Net usada como arquitetura principal. Aceita um ou três canais de saída, conforme a tarefa seja baseline semântico ou segmentação com bordas. |
| `models/segnet.py` | Implementa a SegNet, usada como arquitetura alternativa nas ablações. |
| `losses/cross_entropy.py` | Disponibiliza perdas de entropia cruzada binária e multiclasse ponderada para o treino baseline e para o treino com bordas. |
| `losses/ablation.py` | Implementa `FocalLoss` e a fábrica `create_loss`, usadas para comparar perdas nas experiências de ablação. |
| `metrics/semantic.py` | Calcula Dice e IoU para máscaras semânticas. |
| `metrics/instance.py` | Faz correspondência entre instâncias previstas e reais por IoU; calcula erro de contagem, precisão, recall, F1, AP e mAP. |
| `postprocessing/connected_components.py` | Transforma uma máscara binária do baseline em IDs de instância por componentes conexos. |
| `postprocessing/watershed.py` | Cria marcadores a partir da probabilidade de interior, remove componentes pequenos e aplica Watershed para separar objetos em contato. Aceita probabilidades ou logits da rede. |
| `postprocessing/instance_fusion.py` | Constrói instâncias globais a partir dos tiles sobrepostos e une duplicatas com IoU e Union-Find. Também compara a contagem antes e depois da fusão. |
| `inference/tiled.py` | Carrega a U-Net, prepara imagens, cria posições de tiles, faz padding/predição por tile e executa a inferência em mosaicos. |
| `inference/receptive_field.py` | Estima o campo receptivo da U-Net e mede tamanhos de objetos do dataset para a análise de falhas. |

### Pontos de entrada — `scripts/`

| Arquivo | Como usar | Responsabilidade |
| --- | --- | --- |
| `generate_synthetic_dataset.py` | `python -m scripts.generate_synthetic_dataset` | Gera as imagens sintéticas com elipses e salva a imagem, máscara semântica e máscara de instâncias. Aceita `--output`, `--num-images` e `--seed`. |
| `train.py` | `python -m scripts.train` | Treina a U-Net de três classes com a perda ponderada de bordas e salva os pesos em `checkpoints/unet_boundary.pth`. As configurações ficam no início do arquivo. |
| `evaluate.py` | `python -m scripts.evaluate` | Compara o baseline binário com U-Net + boundary + Watershed. Calcula Dice, IoU, mAP e erro de contagem; grava JSON e gráfico por densidade em `output_dir/evaluation/`. |
| `run_ablations.py` | `python -m scripts.run_ablations` | Treina as configurações de ablação: arquitetura, resolução e função de perda. Salva checkpoints e resultados em `output_dir/ablations/`. |
| `evaluate_ablations.py` | `python -m scripts.evaluate_ablations` | Carrega os resultados/checkpoints das ablações, calcula métricas agregadas e gera gráficos comparativos. |
| `evaluate_tiled.py` | `python -m scripts.evaluate_tiled` | Mede mAP e erro de contagem na inferência por tiles, antes e depois da fusão de instâncias. |
| `visualize_watershed.py` | `python -m scripts.visualize_watershed` | Gera exemplos visuais da predição, marcadores, foreground e segmentação final do Watershed. |
| `visualize_boundary.py` | `python -m scripts.visualize_boundary` | Visualiza a saída da U-Net de bordas junto da imagem e das máscaras de referência. |
| `visualize_tiled.py` | `python -m scripts.visualize_tiled` | Mostra a divisão em tiles, as instâncias previstas por tile e o resultado após a fusão. |
| `failure_analysis.py` | `python -m scripts.failure_analysis` | Localiza amostras com piores resultados, produz uma galeria de falhas e avalia a correção por ajuste do tamanho mínimo dos marcadores. |
| `stress_test.py` | `python -m scripts.stress_test` | Aplica blur, ruído e mudanças de brilho/contraste às imagens e mede a robustez do modelo em cada nível de corrupção. |
| `test_watershed.py` | `python -m scripts.test_watershed` | Teste pequeno e determinístico para verificar se o Watershed separa dois objetos que se tocam. |

### Fluxo recomendado

1. Gere ou atualize os dados com `scripts.generate_synthetic_dataset`.
2. Treine o modelo com `scripts.train`.
3. Avalie a comparação principal com `scripts.evaluate`.
4. Execute, quando necessário, ablações, inferência em mosaicos, visualizações, análise de falhas e teste de robustez.

Os scripts usam caminhos relativos à raiz do repositório. Execute-os sempre a partir
dessa pasta para que `data/`, `checkpoints/` e `output_dir/` sejam encontrados.
