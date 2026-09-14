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