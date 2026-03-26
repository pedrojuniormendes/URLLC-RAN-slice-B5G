# URLLC-RAN-slice-B5G

This repository contains PYTHON scripts for simulating URRLC-RAN-slice-B5G.

## Files

- **URLLC.py** — Simulate the conditions exposed in the article.
- **algorithm.py** — Simulate the algorithm exposed in the article.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/pedrojuniormendes/URLLC-RAN-slice-B5G.git
   cd URLLC-RAN-slice-B5G
   ```

2. Open and set the folder as the current working directory.
   ```
   Was tested on Spyder.
   ```
## Algoritmo 1
## 📡 Algoritmo — Orquestração de Recursos para URLLC (SNC)

**Entrada:**
- $|\mathcal{I}|$: número de células  
- $|\mathcal{M}|$: número de fatias  
- $\Delta T$: janela temporal  
- $(t_{slot}, \theta, \delta)$: parâmetros SNC  
- $(W_m, \epsilon'_m)$: requisitos por fatia  
- $\{(c_q, p_q)\}$: PMF de capacidade  
- $\zeta_{user}$: demanda  

**Saída:**
- Alocação de recursos (RBs tempo × frequência)

---

```text
WHILE rede ativa DO
    FOR EACH slice m ∈ M DO
        FOR EACH célula i ∈ I DO

            Atualizar estatísticas do canal
            Construir Q_i^m e PMF {(c_q, p_q)}

            Calcular W_i,m (SNC)
            Estimar N_max_i,m

            Medir N_cur_i,m

            IF N_cur_i,m ≤ N_max_i,m THEN
                Manter recursos
            ELSE
                Aumentar RBs
            END IF

        END FOR
    END FOR

    Esperar ΔT
END WHILE

## Contact

Author: Pedro Mendes da Silva Júnior 
GitHub: [pedrojuniormendes](https://github.com/pedrojuniormendes)

Author: Flávio Geraldo Coelho Rocha  GitHub: [flaviogeraldo](https://github.com/flaviogeraldo)








