import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

num_celulas = 5
num_slices = 3
delta_t = 1.0

larguras_banda = [10, 10, 15, 15, 20]
frequencia_portadora = 3.55
raio_celula = 300
potencia_transmissao = 33

parametros_snc = {
    't_slot': 0.1,
    'theta': 0.1,
    'delta': 0.01
}

slices = {
    0: {'nome': 'URLLC', 'W_m': 0.01, 'epsilon_m': 1e-5, 'zeta_user': 1.0},
    1: {'nome': 'eMBB', 'W_m': 0.1, 'epsilon_m': 1e-3, 'zeta_user': 5.0},
    2: {'nome': 'mMTC', 'W_m': 1.0, 'epsilon_m': 1e-2, 'zeta_user': 0.1}
}

estado_celulas = {}
historico_orquestracao = []
historico_viabilidade = []

def inicializar_estado_rede():
    global estado_celulas
    for i in range(num_celulas):
        estado_celulas[i] = {
            'usuarios_ativos': defaultdict(int),
            'recursos_alocados': defaultdict(float),
            'capacidade_total': calcular_capacidade_teorica(i),
            'estatisticas_canal': gerar_estatisticas_canal(),
            'pmf_capacidade': gerar_pmf_capacidade(i)
        }

def calcular_capacidade_teorica(id_celula):
    largura_banda = larguras_banda[id_celula]
    eficiencia_espectral = 3.5
    return largura_banda * eficiencia_espectral

def gerar_estatisticas_canal():
    return {
        'snr_medio': np.random.uniform(10, 20),
        'variancia_snr': np.random.uniform(1, 5),
        'taxa_los': np.random.uniform(0.3, 0.9),
        'interferencia_media': np.random.uniform(3, 8)
    }

def gerar_pmf_capacidade(id_celula):
    capacidade_media = calcular_capacidade_teorica(id_celula)
    desvio = capacidade_media * 0.2
    niveis_q = 5
    pmf = []
    capacidades = np.linspace(capacidade_media - desvio, capacidade_media + desvio, niveis_q)
    probabilidades = np.random.dirichlet(np.ones(niveis_q))
    for c, p in zip(capacidades, probabilidades):
        pmf.append((max(c, 0.1), p))
    return pmf

def calcular_delay_bound_snc(pmf_capacidade, theta, t_slot):
    soma_exp = 0.0
    for c_q, p_q in pmf_capacidade:
        if c_q > 0:
            soma_exp += p_q * np.exp(-theta * c_q * t_slot)
    if soma_exp <= 0:
        return float('inf')
    W_bound = -np.log(soma_exp) / theta
    return W_bound

def calcular_n_max_usuarios(W_m, epsilon_m, pmf_capacidade, theta, delta, t_slot, zeta_user):
    if len(pmf_capacidade) == 0:
        return 0
    
    W_bound = calcular_delay_bound_snc(pmf_capacidade, theta, t_slot)
    if W_bound == float('inf'):
        return 0
    
    prob_violacao = np.exp(-theta * W_bound)
    if prob_violacao <= 0:
        return 0
    
    try:
        n_max = int(-np.log(epsilon_m) / (theta * zeta_user * prob_violacao))
    except:
        n_max = 0
    
    if W_bound > W_m:
        fator_reducao = W_m / max(W_bound, 1e-10)
        n_max = int(n_max * fator_reducao)
    
    return max(0, n_max)

def atualizar_estatisticas_canal(id_celula):
    estatisticas = estado_celulas[id_celula]['estatisticas_canal']
    estatisticas['snr_medio'] += np.random.normal(0, 0.5)
    estatisticas['snr_medio'] = np.clip(estatisticas['snr_medio'], 5, 25)
    estatisticas['taxa_los'] += np.random.normal(0, 0.05)
    estatisticas['taxa_los'] = np.clip(estatisticas['taxa_los'], 0.1, 1.0)
    estado_celulas[id_celula]['pmf_capacidade'] = gerar_pmf_capacidade(id_celula)

def simular_carga_usuarios():
    for id_celula in range(num_celulas):
        for slice_id in range(num_slices):
            chegada = np.random.poisson(slices[slice_id]['zeta_user'] * 0.5)
            usuarios_ativos = estado_celulas[id_celula]['usuarios_ativos'][slice_id]
            saida = np.random.binomial(usuarios_ativos, 0.1)
            novo_total = max(0, usuarios_ativos + chegada - saida)
            estado_celulas[id_celula]['usuarios_ativos'][slice_id] = novo_total

def executar_orquestracao():
    resultados_iteracao = {
        'celulas': {},
        'recursos_ajustados': False
    }
    
    for slice_id in range(num_slices):
        slice_info = slices[slice_id]
        W_m = slice_info['W_m']
        epsilon_m = slice_info['epsilon_m']
        zeta_user = slice_info['zeta_user']
        
        for id_celula in range(num_celulas):
            pmf_capacidade = estado_celulas[id_celula]['pmf_capacidade']
            
            n_max = calcular_n_max_usuarios(
                W_m, epsilon_m, pmf_capacidade,
                parametros_snc['theta'], parametros_snc['delta'],
                parametros_snc['t_slot'], zeta_user
            )
            
            n_atual = estado_celulas[id_celula]['usuarios_ativos'][slice_id]
            viavel = n_atual <= n_max
            
            capacidade_celula = estado_celulas[id_celula]['capacidade_total']
            
            if not viavel:
                fator_aumento = min(2.0, n_atual / max(n_max, 1))
                recursos_necessarios = capacidade_celula * fator_aumento * 0.3
                resultados_iteracao['recursos_ajustados'] = True
                estado_celulas[id_celula]['recursos_alocados'][slice_id] = recursos_necessarios
                
                capacidade_restante = capacidade_celula - recursos_necessarios
                for other_slice in range(num_slices):
                    if other_slice != slice_id:
                        proporcao_atual = estado_celulas[id_celula]['recursos_alocados'].get(other_slice, 0)
                        proporcao_ajustada = proporcao_atual * (capacidade_restante / capacidade_celula)
                        estado_celulas[id_celula]['recursos_alocados'][other_slice] = proporcao_ajustada
            else:
                proporcao = n_atual / max(n_max, 1)
                recursos_necessarios = capacidade_celula * proporcao * 0.25
                estado_celulas[id_celula]['recursos_alocados'][slice_id] = recursos_necessarios
            
            if id_celula not in resultados_iteracao['celulas']:
                resultados_iteracao['celulas'][id_celula] = {}
            
            resultados_iteracao['celulas'][id_celula][slice_id] = {
                'n_max': n_max,
                'n_atual': n_atual,
                'viavel': viavel,
                'recursos_alocados': estado_celulas[id_celula]['recursos_alocados'][slice_id],
                'W_bound': calcular_delay_bound_snc(pmf_capacidade, parametros_snc['theta'], parametros_snc['t_slot'])
            }
    
    return resultados_iteracao

def plotar_resultados():
    if len(historico_viabilidade) < 2:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    iteracoes = range(len(historico_viabilidade))
    axes[0,0].plot(iteracoes, historico_viabilidade, 'b-', linewidth=2)
    axes[0,0].set_xlabel('Iteração')
    axes[0,0].set_ylabel('Viabilidade URLLC (%)')
    axes[0,0].set_title('Evolução da Viabilidade URLLC')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim([0, 100])
    
    usuarios_por_slice = []
    nomes_slices = []
    
    for slice_id in range(num_slices):
        slice_nome = slices[slice_id]['nome']
        usuarios_total = 0
        for resultado in historico_orquestracao:
            for id_celula in resultado['celulas']:
                if slice_id in resultado['celulas'][id_celula]:
                    usuarios_total += resultado['celulas'][id_celula][slice_id]['n_atual']
        usuarios_por_slice.append(usuarios_total)
        nomes_slices.append(slice_nome)
    
    axes[0,1].bar(nomes_slices, usuarios_por_slice, alpha=0.7)
    axes[0,1].set_xlabel('Slice')
    axes[0,1].set_ylabel('Total de Usuários')
    axes[0,1].set_title('Distribuição de Usuários por Slice')
    axes[0,1].grid(True, alpha=0.3)
    
    recursos_por_celula = []
    labels_celulas = []
    
    for id_celula in range(num_celulas):
        largura_banda = larguras_banda[id_celula]
        recursos_total = 0
        for resultado in historico_orquestracao:
            if id_celula in resultado['celulas']:
                for slice_id in resultado['celulas'][id_celula]:
                    recursos_total += resultado['celulas'][id_celula][slice_id]['recursos_alocados']
        recursos_por_celula.append(recursos_total)
        labels_celulas.append(f'Célula {id_celula}\n({largura_banda} MHz)')
    
    axes[0,2].bar(labels_celulas, recursos_por_celula, alpha=0.7)
    axes[0,2].set_xlabel('Célula')
    axes[0,2].set_ylabel('Recursos Alocados Totais (Mbps)')
    axes[0,2].set_title('Recursos Alocados por Célula')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].tick_params(axis='x', rotation=45)
    
    delay_bounds = defaultdict(list)
    for resultado in historico_orquestracao:
        for id_celula in resultado['celulas']:
            for slice_id in resultado['celulas'][id_celula]:
                delay_bounds[slice_id].append(resultado['celulas'][id_celula][slice_id]['W_bound'])
    
    for slice_id in range(num_slices):
        slice_nome = slices[slice_id]['nome']
        W_m = slices[slice_id]['W_m']
        if delay_bounds[slice_id]:
            axes[1,0].plot(iteracoes[:len(delay_bounds[slice_id])], 
                         delay_bounds[slice_id][:len(iteracoes)], 
                         label=f'{slice_nome} (W_m={W_m}s)', linewidth=2)
    
    axes[1,0].set_xlabel('Iteração')
    axes[1,0].set_ylabel('Delay Bound (s)')
    axes[1,0].set_title('Delay Bounds Calculados por SNC')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    viabilidade_por_slice = defaultdict(list)
    for resultado in historico_orquestracao:
        for id_celula in resultado['celulas']:
            for slice_id in resultado['celulas'][id_celula]:
                viabilidade_por_slice[slice_id].append(1 if resultado['celulas'][id_celula][slice_id]['viavel'] else 0)
    
    medias_viabilidade = []
    desvios_viabilidade = []
    for slice_id in range(num_slices):
        if viabilidade_por_slice[slice_id]:
            media = np.mean(viabilidade_por_slice[slice_id]) * 100
            desvio = np.std(viabilidade_por_slice[slice_id]) * 100
            medias_viabilidade.append(media)
            desvios_viabilidade.append(desvio)
    
    if medias_viabilidade:
        x_pos = np.arange(len(nomes_slices))
        axes[1,1].bar(x_pos, medias_viabilidade, yerr=desvios_viabilidade, capsize=5, alpha=0.7)
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(nomes_slices)
        axes[1,1].set_ylabel('Viabilidade (%)')
        axes[1,1].set_title('Viabilidade por Slice')
        axes[1,1].grid(True, alpha=0.3)
    
    eficiencia_celulas = []
    for id_celula in range(num_celulas):
        capacidade_total = estado_celulas[id_celula]['capacidade_total']
        utilizacao_total = 0
        contagem = 0
        for resultado in historico_orquestracao:
            if id_celula in resultado['celulas']:
                for slice_id in resultado['celulas'][id_celula]:
                    utilizacao_total += resultado['celulas'][id_celula][slice_id]['recursos_alocados']
                    contagem += 1
        if contagem > 0:
            utilizacao_media = utilizacao_total / contagem
            eficiencia = 100 * utilizacao_media / capacidade_total
            eficiencia_celulas.append(eficiencia)
    
    if eficiencia_celulas:
        x_pos = np.arange(len(eficiencia_celulas))
        axes[1,2].bar(x_pos, eficiencia_celulas, alpha=0.7)
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels([f'Célula {i}' for i in range(len(eficiencia_celulas))])
        axes[1,2].set_ylabel('Eficiência de Utilização (%)')
        axes[1,2].set_title('Eficiência por Célula')
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def executar_simulacao(num_iteracoes=50):
    global historico_orquestracao, historico_viabilidade
    
    print("="*70)
    print("SIMULAÇÃO DE ORQUESTRAÇÃO URLLC USANDO SNC")
    print("="*70)
    
    inicializar_estado_rede()
    
    for iteracao in range(num_iteracoes):
        print(f"\nIteração {iteracao + 1}/{num_iteracoes}")
        print("-"*50)
        
        for id_celula in range(num_celulas):
            atualizar_estatisticas_canal(id_celula)
        
        simular_carga_usuarios()
        
        resultados = executar_orquestracao()
        historico_orquestracao.append(resultados)
        
        viabilidade_total = 0
        total_conexoes = 0
        
        for id_celula in resultados['celulas']:
            for slice_id in resultados['celulas'][id_celula]:
                if resultados['celulas'][id_celula][slice_id]['viavel']:
                    viabilidade_total += 1
                total_conexoes += 1
        
        taxa_viabilidade = 100 * viabilidade_total / max(total_conexoes, 1)
        historico_viabilidade.append(taxa_viabilidade)
        
        print(f"Taxa de viabilidade URLLC: {taxa_viabilidade:.1f}%")
        
        if resultados['recursos_ajustados']:
            print("Ajustes de recursos realizados")
        
        if 0 in resultados['celulas']:
            print("\nDetalhes da Célula 0:")
            for slice_id in resultados['celulas'][0]:
                slice_nome = slices[slice_id]['nome']
                dados = resultados['celulas'][0][slice_id]
                status = "VIÁVEL" if dados['viavel'] else "NÃO VIÁVEL"
                print(f"  {slice_nome}: {dados['n_atual']}/{dados['n_max']} usuários - {status}")
    
    analisar_resultados()

def analisar_resultados():
    print("\n" + "="*70)
    print("ANÁLISE FINAL DA ORQUESTRAÇÃO SNC")
    print("="*70)
    
    if not historico_orquestracao:
        print("Nenhum resultado para analisar!")
        return
    
    viabilidade_media = np.mean(historico_viabilidade)
    viabilidade_std = np.std(historico_viabilidade)
    
    print(f"\nDesempenho Geral:")
    print(f"  Viabilidade média: {viabilidade_media:.1f}%")
    print(f"  Desvio padrão: {viabilidade_std:.1f}%")
    
    estatisticas_slice = defaultdict(lambda: {'viabilidade': [], 'usuarios': [], 'recursos': []})
    
    for resultado in historico_orquestracao:
        for id_celula in resultado['celulas']:
            for slice_id in resultado['celulas'][id_celula]:
                dados = resultado['celulas'][id_celula][slice_id]
                estatisticas_slice[slice_id]['viabilidade'].append(1 if dados['viavel'] else 0)
                estatisticas_slice[slice_id]['usuarios'].append(dados['n_atual'])
                estatisticas_slice[slice_id]['recursos'].append(dados['recursos_alocados'])
    
    for slice_id in range(num_slices):
        slice_nome = slices[slice_id]['nome']
        viabilidade_slice = np.mean(estatisticas_slice[slice_id]['viabilidade']) * 100
        usuarios_medio = np.mean(estatisticas_slice[slice_id]['usuarios'])
        recursos_medio = np.mean(estatisticas_slice[slice_id]['recursos'])
        
        print(f"\n  {slice_nome}:")
        print(f"    Viabilidade: {viabilidade_slice:.1f}%")
        print(f"    Usuários médios: {usuarios_medio:.1f}")
        print(f"    Recursos alocados médios: {recursos_medio:.1f} Mbps")
    
    for id_celula in range(num_celulas):
        largura_banda = larguras_banda[id_celula]
        capacidade_total = estado_celulas[id_celula]['capacidade_total']
        utilizacao_total = 0
        contagem = 0
        
        for resultado in historico_orquestracao:
            if id_celula in resultado['celulas']:
                for slice_id in resultado['celulas'][id_celula]:
                    utilizacao_total += resultado['celulas'][id_celula][slice_id]['recursos_alocados']
                    contagem += 1
        
        utilizacao_media = utilizacao_total / max(contagem, 1)
        taxa_utilizacao = 100 * utilizacao_media / capacidade_total
        
        print(f"\n  Célula {id_celula} ({largura_banda} MHz):")
        print(f"    Capacidade total: {capacidade_total:.1f} Mbps")
        print(f"    Utilização média: {utilizacao_media:.1f} Mbps ({taxa_utilizacao:.1f}%)")
    
    plotar_resultados()

def main():
    executar_simulacao(num_iteracoes=30)

if __name__ == "__main__":
    main()