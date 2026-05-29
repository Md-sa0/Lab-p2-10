# Laboratório 10: O Pipeline Definitivo
## RAG + QLoRA + FlashAttention-2 + KV Cache

> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por [Seu Nome]**

---

## Sumário

- [Descrição do Problema](#descrição-do-problema)
- [Arquitetura do Pipeline](#arquitetura-do-pipeline)
- [Como Executar](#como-executar)
- [Métricas de Benchmark](#métricas-de-benchmark)
- [Análise Arquitetural (Passo 5)](#análise-arquitetural-passo-5)
- [Estrutura do Repositório](#estrutura-do-repositório)

---

## Descrição do Problema

A **HealthTech** precisa de um pipeline de geração automática de relatórios clínicos. O fluxo:

1. Um **RAG** recupera ~5 capítulos de manuais médicos (~12.000 tokens)
2. Um modelo **Llama** fine-tunado em jargão médico (via QLoRA) processa o contexto
3. O modelo gera um **resumo clínico** estruturado

**O desastre:** a complexidade **O(n²)** do Self-Attention padrão estourou a VRAM da GPU ao processar o contexto massivo. Este laboratório resolve o problema com três otimizações combinadas.

---

## Arquitetura do Pipeline

```
[PDFs Médicos] → [RAG / Banco Vetorial] → [Contexto ~12k tokens]
                                                    ↓
                              [TinyLlama 4-bits (QLoRA via bitsandbytes)]
                                    ↓                    ↓
                            [FlashAttention-2]      [KV Cache]
                            (Hardware-Aware,        (Software-Aware,
                             SRAM da GPU)            evita recálculo Q,K,V)
                                                    ↓
                                         [Resumo Clínico Gerado]
```

---

## Como Executar

### Requisitos

```bash
pip install transformers>=4.40.0 bitsandbytes>=0.43.0 accelerate>=0.29.0
pip install flash-attn --no-build-isolation   # Requer GPU Ampere+ (RTX 3090, A100 etc.)
```

### Executar o Notebook

```bash
jupyter notebook lab10_pipeline_definitivo.ipynb
```

> **Nota:** É obrigatório ter GPU com suporte a CUDA. FlashAttention-2 requer arquitetura Ampere ou superior.

---

## Métricas de Benchmark

> Os valores abaixo são **referência**. Os valores reais serão preenchidos após execução na GPU.

| Configuração | Tempo Geração (100 tok) | Pico VRAM | Throughput |
|---|---|---|---|
| **Baseline** (sem KV Cache) | ~Xs | ~X MB | ~X tok/s |
| **Otimizado** (KV Cache + FlashAttn-2) | ~Xs | ~X MB | ~X tok/s |
| **Redução** | **-XX%** | **-XX%** | **+XXx** |

### Passo 1 — VRAM do modelo carregado em 4-bits (QLoRA)

| Modelo | Precisão | VRAM |
|---|---|---|
| TinyLlama-1.1B | Float16 (baseline) | ~2.200 MB |
| TinyLlama-1.1B | **4-bits (QLoRA)** | **~700–800 MB** |

*Valores reais a serem registrados após execução.*

---

## Análise Arquitetural (Passo 5)

### Parte A — Como QLoRA + KV Cache + FlashAttention salvaram o pipeline

O colapso de VRAM neste laboratório tem três causas encadeadas: (1) o modelo base em Float16 já ocuparia ~2 GB apenas para seus pesos; (2) o Self-Attention padrão aloca uma matriz de atenção de tamanho O(n²) na VRAM — com 12.000 tokens de contexto, isso representa ~576 milhões de posições de atenção por camada; (3) sem KV Cache, os vetores **Key** e **Value** de todos os tokens anteriores são recalculados do zero a cada novo token gerado, multiplicando o custo computacional de forma quadrática. A combinação das três otimizações desfaz exatamente essas três causas: o **QLoRA em 4-bits** (via `bitsandbytes`) comprime os pesos do modelo de ~2 GB para ~700–800 MB, liberando a maior parte da VRAM antes mesmo da inferência começar; o **KV Cache** (`use_cache=True`) armazena os vetores K e V já calculados e os reutiliza a cada passo autoregressivo, reduzindo a complexidade de geração de O(n²·T) para O(n + T), onde T é o número de novos tokens; e o **FlashAttention-2** (`attn_implementation="flash_attention_2"`) reestrutura o cálculo da atenção em blocos que cabem na SRAM on-chip da GPU, evitando a leitura e escrita repetida da matriz de atenção na VRAM (HBM), o que reduz drasticamente os acessos à memória sem alterar matematicamente o resultado. Em conjunto, as três técnicas transformam um pipeline que travava a GPU em um sistema viável de produção.

### Parte B — Por que 2 milhões de tokens ainda quebraria tudo, e por que Mamba é a resposta

Mesmo com FlashAttention-2, o Transformer continua sendo fundamentalmente um modelo de atenção **full-quadrática sobre todos os tokens da sequência de entrada** — o FlashAttention apenas torna esse O(n²) mais eficiente em termos de acesso à memória, mas não elimina a complexidade assintótica. Com 2 milhões de tokens, a matriz de atenção teria 4 trilhões de posições (2.000.000²); o KV Cache, por sua vez, precisaria armazenar dois vetores (K e V) para cada um dos 2 milhões de tokens em cada camada e cada cabeça de atenção — para um modelo de porte razoável, isso ultrapassaria dezenas de gigabytes apenas de cache. O FlashAttention-2 reduziria os acessos à VRAM por bloco, mas a quantidade total de computação e o tamanho do KV Cache cresceriam de forma insustentável, estourando até GPUs com 80 GB como a A100. É exatamente para esse regime de contextos ultra-longos que a arquitetura **Mamba** (e os **State Space Models** em geral) se torna a solução necessária: em vez de reprocessar toda a sequência passada via atenção, o Mamba mantém um **estado oculto comprimido de dimensão fixa**, independentemente do comprimento da sequência — o que lhe confere complexidade de memória **O(1)** e complexidade temporal **O(n)** durante a inferência. Para o caso da HealthTech com 2 milhões de tokens, a migração para Mamba não seria uma otimização opcional, mas uma exigência arquitetural: somente um modelo com memória constante poderia processar esse volume sem explodir a VRAM, mesmo em hardware de alto desempenho.

---

## Estrutura do Repositório

```
lab10-pipeline-definitivo/
├── lab10_pipeline_definitivo.ipynb   # Notebook principal com os 4 passos
├── README.md                         # Este arquivo (relatório + análise)
└── benchmark_lab10.png               # Gráfico comparativo gerado automaticamente
```

---

## Referências

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- Hugging Face. *BitsAndBytes Integration*. https://huggingface.co/docs/transformers/quantization/bitsandbytes
