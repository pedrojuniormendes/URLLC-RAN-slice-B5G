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
```latex
\begin{algorithm}[htbp]
\caption{Orquestração de recursos para fatia URLLC usando \ac{SNC}}
\label{alg:snc_orchestration}
\DontPrintSemicolon

\KwIn{
Qtde de células $|\mathcal{I}|$; Qtde de fatias $|\mathcal{M}|$; janela temporal $\Delta T$; 
parâmetros do \ac{SNC} $t_{\mathrm{slot}},\theta,\delta$; requisitos por fatia $(W_m,\epsilon'_m)$; 
PMF de capacidade $\{(c_q,p_q)\}_{q\in\mathcal{Q}_{i}^{m}}$; 
demanda $\zeta_{\mathrm{user}}$.
}
\KwOut{Distribuição de recursos (RBs tempo$\times$frequência).}

\While{rede ativa}{
  \ForEach{slice $m\in\mathcal{M}$}{
    \ForEach{célula $i\in\mathcal{I}$}{
      Atualizar estatísticas do canal\;
      Calcular $W_{i,m}$\;
      Estimar $N^{\max}_{i,m}$\;
      Medir $N^{\mathrm{cur}}_{i,m}$\;

      \eIf{$N^{\mathrm{cur}}_{i,m} \le N^{\max}_{i,m}$}{
        Manter recursos\;
      }{
        Aumentar recursos\;
      }
    }
  }
  Esperar $\Delta T$\;
}
\end{algorithm}
## Contact

Author: Pedro Mendes da Silva Júnior 
GitHub: [pedrojuniormendes](https://github.com/pedrojuniormendes)

Author: Flávio Geraldo Coelho Rocha  GitHub: [flaviogeraldo](https://github.com/flaviogeraldo)








