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

\BlankLine

\While{rede ativa}{
  \ForEach{slice $m\in\mathcal{M}$}{
    \ForEach{célula $i\in\mathcal{I}$}{
      Obter/atualizar estatísticas do canal e da capacidade para formar $\mathcal{Q}_{i}^{m}$ e a PMF $\{(c_q,p_q)\}$\;
      
      Calcular o \emph{delay bound} $W_{i,m}$ baseado em \ac{SNC} utilizando a expressão fechada dada em (\ref{eq:wi_m});
      Usar a relação analítica (\ref{eq:n_user_final}) fechada para estimar o número máximo de usuários admitidos
      $N^{\max}_{i,m}$ em função de $(W_m,\epsilon'_m)$, $\theta$, $\delta$, $t_{\mathrm{slot}}$, $p_q$ e $|\mathcal{I}|$\;
      
      Medir o número atual de usuários conectados na célula $N^{\mathrm{cur}}_{i,m}$\;
      
      \eIf{$N^{\mathrm{cur}}_{i,m} \le N^{\max}_{i,m}$}{
        Manter a cota de recursos da slice $m$ em $i$\;
      }{
        Aumentar a cota de recursos (RBs) da slice $m$ em $i$ para restaurar viabilidade\;
      }
    }
  }
  
  % \BlankLine
  % Calcular a capacidade total prevista do orquestrador para o cenário $k$:
  % $N(k) = \sum_{m\in\mathcal{M}} n_{k,m}$\;
  
  % \ForEach{slice $m\in\mathcal{M}$}{
  %   Definir a proporção/cota global da slice $m$ a partir da capacidade prevista, por exemplo:
  %   $\eta_{k,m} = \dfrac{n_{k,m}}{N(k)}$\;
  %   Aplicar $\eta_{k,m}$ como regra de partição global e projetar para cada célula)\;
  % }
  
  Esperar $\Delta T$ e repetir\;
}
\end{algorithm}
## Contact

Author: Pedro Mendes da Silva Júnior 
GitHub: [pedrojuniormendes](https://github.com/pedrojuniormendes)

Author: Flávio Geraldo Coelho Rocha  GitHub: [flaviogeraldo](https://github.com/flaviogeraldo)








