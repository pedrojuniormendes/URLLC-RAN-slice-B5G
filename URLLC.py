import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

NUM_CELULAS = 5
LARGURAS_BANDA = [10, 10, 15, 15, 20]
FREQUENCIA_PORTADORA = 3.55
RAIO_CELULA = 300
POTENCIA_TRANSMISSAO = 33

VARIAVEIS_AMBIENTE = {
    'condicoes_climaticas': ['ceu_limpo', 'chuva', 'neblina', 'tempestade'],
    'densidade_urbana': ['densa_urbana', 'urbana', 'suburbana', 'rural'],
    'periodo_dia': ['pico', 'fora_pico', 'noite'],
    'padroes_mobilidade': ['estatico', 'pedestre', 'veicular']
}

PROBABILIDADES_EVENTOS = {
    'handover': 0.1,
    'pico_interferencia': 0.05,
    'variacao_canal': 0.3,
    'violacao_qos': 0.15
}

def obter_fatores_ambientais(iteracao, total_iteracoes):
    fator_tempo = (iteracao % 24) / 24
    probabilidades_clima = [0.6, 0.2, 0.15, 0.05]
    clima = np.random.choice(VARIAVEIS_AMBIENTE['condicoes_climaticas'], p=probabilidades_clima)
    
    if fator_tempo < 0.25 or fator_tempo > 0.75:
        densidade_urbana = 'rural'
    elif 0.4 < fator_tempo < 0.6:
        densidade_urbana = 'densa_urbana'
    else:
        densidade_urbana = np.random.choice(['urbana', 'suburbana'], p=[0.6, 0.4])
    
    if fator_tempo < 0.25:
        periodo_dia = 'noite'
    elif 0.25 <= fator_tempo < 0.5:
        periodo_dia = 'fora_pico'
    else:
        periodo_dia = 'pico'
    
    if densidade_urbana == 'densa_urbana':
        mobilidade = np.random.choice(['pedestre', 'veicular'], p=[0.7, 0.3])
    else:
        mobilidade = np.random.choice(['estatico', 'pedestre', 'veicular'], p=[0.3, 0.4, 0.3])
    
    return {
        'clima': clima,
        'densidade_urbana': densidade_urbana,
        'periodo_dia': periodo_dia,
        'mobilidade': mobilidade,
        'fator_tempo': fator_tempo,
        'iteracao': iteracao
    }

def aplicar_efeitos_climaticos(clima, parametros_base):
    efeitos_clima = {
        'ceu_limpo': {'deslocamento_perda_percurso': 0, 'desvio_sombreamento': 3.0, 'fator_disponibilidade': 1.0},
        'chuva': {'deslocamento_perda_percurso': 2.5, 'desvio_sombreamento': 4.0, 'fator_disponibilidade': 0.95},
        'neblina': {'deslocamento_perda_percurso': 1.5, 'desvio_sombreamento': 3.5, 'fator_disponibilidade': 0.98},
        'tempestade': {'deslocamento_perda_percurso': 5.0, 'desvio_sombreamento': 6.0, 'fator_disponibilidade': 0.85}
    }
    
    efeitos = efeitos_clima[clima].copy()
    for chave in efeitos:
        if chave != 'fator_disponibilidade':
            efeitos[chave] += np.random.normal(0, efeitos[chave] * 0.1)
    
    return {**parametros_base, **efeitos}

def aplicar_efeitos_densidade_urbana(densidade, parametros_base):
    efeitos_densidade = {
        'densa_urbana': {
            'probabilidade_los': 0.3,
            'nivel_interferencia': 8.0,
            'media_perda_edificios': 20.0
        },
        'urbana': {
            'probabilidade_los': 0.5,
            'nivel_interferencia': 5.0,
            'media_perda_edificios': 15.0
        },
        'suburbana': {
            'probabilidade_los': 0.7,
            'nivel_interferencia': 2.0,
            'media_perda_edificios': 10.0
        },
        'rural': {
            'probabilidade_los': 0.9,
            'nivel_interferencia': 1.0,
            'media_perda_edificios': 5.0
        }
    }
    
    efeitos = efeitos_densidade[densidade].copy()
    for chave in efeitos:
        efeitos[chave] *= np.random.uniform(0.9, 1.1)
    
    return {**parametros_base, **efeitos}

def aplicar_efeitos_mobilidade(mobilidade, parametros_base, velocidade_ue=None):
    if velocidade_ue is None:
        velocidades_mobilidade = {
            'estatico': (0, 0.5),
            'pedestre': (0.5, 5),
            'veicular': (5, 30)
        }
        velocidade_min, velocidade_max = velocidades_mobilidade[mobilidade]
        velocidade_ue = np.random.uniform(velocidade_min, velocidade_max)
    
    efeitos_mobilidade = {
        'espalhamento_doppler': velocidade_ue * FREQUENCIA_PORTADORA * 1e9 / 3e8,
        'probabilidade_handover': min(0.05 + velocidade_ue/100, 0.3),
        'tempo_coerencia_canal': 0.423 / (velocidade_ue * FREQUENCIA_PORTADORA / 3e8) if velocidade_ue > 0 else float('inf')
    }
    
    return {**parametros_base, **efeitos_mobilidade, 'velocidade_ue': velocidade_ue}

def calcular_probabilidade_los(distancia_2d, densidade_urbana):
    probabilidades_base = {
        'densa_urbana': 0.3,
        'urbana': 0.5,
        'suburbana': 0.7,
        'rural': 0.9
    }
    
    prob_base = probabilidades_base[densidade_urbana]
    if distancia_2d <= 18:
        fator_distancia = 1.0
    else:
        fator_distancia = 18 / distancia_2d
    
    variacao_aleatoria = np.random.uniform(0.9, 1.1)
    return min(prob_base * fator_distancia * variacao_aleatoria, 1.0)

def calcular_perda_percurso_3gpp(distancia_2d, distancia_3d, condicao_los, parametros_ambientais):
    fc = FREQUENCIA_PORTADORA
    h_BS = 10.0
    h_UT = 1.5
    
    deslocamento_clima = parametros_ambientais.get('deslocamento_perda_percurso', 0)
    perda_edificios = np.random.normal(
        parametros_ambientais.get('media_perda_edificios', 10),
        parametros_ambientais.get('desvio_perda_edificios', 3)
    )
    
    if condicao_los == 'LOS':
        if distancia_2d <= 5 * h_BS * h_UT * fc / 3:
            pl = 21 * np.log10(distancia_3d) + 20 * np.log10(fc) + 32.9
        else:
            pl = 40 * np.log10(distancia_3d) + 20 * np.log10(fc) - 9.5
    else:
        pl_los = 21 * np.log10(distancia_3d) + 20 * np.log10(fc) + 32.9
        pl_nlos = 35.3 * np.log10(distancia_3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (h_UT - 1.5)
        pl = max(pl_los, pl_nlos)
    
    pl_total = pl + deslocamento_clima + perda_edificios
    pl_total += np.random.normal(0, parametros_ambientais.get('variacao_perda_percurso', 1.5))
    return max(pl_total, 60)

def calcular_sombreamento(condicao_los, parametros_ambientais):
    sigma_base = 3.0 if condicao_los == 'LOS' else 4.0
    fator_urbano = {
        'densa_urbana': 1.5,
        'urbana': 1.2,
        'suburbana': 1.0,
        'rural': 0.8
    }[parametros_ambientais.get('densidade_urbana', 'urbana')]
    
    sigma = sigma_base * fator_urbano * np.random.uniform(0.9, 1.1)
    if parametros_ambientais.get('clima') in ['chuva', 'tempestade']:
        sigma *= 1.3
    
    return np.random.normal(0, sigma)

def gerar_amostras_desvanecimento_rapido(num_amostras, parametros_ambientais):
    velocidade = parametros_ambientais.get('velocidade_ue', 1.0)
    frequencia_doppler = velocidade * FREQUENCIA_PORTADORA * 1e9 / 3e8
    prob_los = parametros_ambientais.get('probabilidade_los', 0.5)
    
    if np.random.random() < prob_los:
        k_db = np.random.uniform(7, 13)
    else:
        k_db = -np.inf
    
    fator_k = 10 ** (k_db / 10) if k_db > -np.inf else 0
    t = np.linspace(0, 1, num_amostras)
    
    if fator_k > 0:
        fase_los = 2 * np.pi * frequencia_doppler * t + np.random.uniform(0, 2*np.pi)
        componente_los = np.sqrt(fator_k/(fator_k + 1)) * np.exp(1j * fase_los)
    else:
        componente_los = 0
    
    nlos_complexo = (np.random.randn(num_amostras) + 1j * np.random.randn(num_amostras)) / np.sqrt(2)
    
    if frequencia_doppler > 0 and num_amostras > 1:
        tempo_correlacao = min(0.1 / frequencia_doppler, 0.5)
        alpha = np.exp(-1 / (tempo_correlacao * num_amostras))
        nlos_filtrado = np.zeros_like(nlos_complexo, dtype=complex)
        nlos_filtrado[0] = nlos_complexo[0]
        for i in range(1, num_amostras):
            nlos_filtrado[i] = alpha * nlos_filtrado[i-1] + np.sqrt(1-alpha**2) * nlos_complexo[i]
        componente_nlos = np.sqrt(1/(fator_k + 1)) * nlos_filtrado
    else:
        componente_nlos = np.sqrt(1/(fator_k + 1)) * nlos_complexo
    
    canal = componente_los + componente_nlos
    return 20 * np.log10(np.abs(canal) + 1e-10)

def calcular_capacidade_shannon(sinr_db, largura_banda_mhz):
    sinr_linear = 10**(sinr_db/10)
    eficiencia_espectral = np.log2(1 + sinr_linear)
    largura_banda_hz = largura_banda_mhz * 1e6
    capacidade_mbps = (largura_banda_hz * eficiencia_espectral) / 1e6
    return capacidade_mbps

def gerar_trafego_background_dinamico(id_celula, iteracao, parametros_ambientais):
    capacidade_base = {10: 50, 15: 75, 20: 100}
    bw = LARGURAS_BANDA[id_celula]
    capacidade_maxima = capacidade_base[bw]
    fator_tempo = parametros_ambientais.get('fator_tempo', 0.5)
    multiplicador_tempo = 0.3 + 0.7 * (1 - np.cos(2 * np.pi * fator_tempo)) / 2
    
    multiplicador_densidade = {
        'densa_urbana': 1.3,
        'urbana': 1.1,
        'suburbana': 0.9,
        'rural': 0.7
    }[parametros_ambientais.get('densidade_urbana', 'urbana')]
    
    demanda_base = np.random.uniform(0.5, 0.75) * capacidade_maxima
    demanda_ajustada = demanda_base * multiplicador_tempo * multiplicador_densidade
    flutuacao_aleatoria = np.random.normal(1, 0.1)
    demanda_ajustada *= flutuacao_aleatoria
    
    demanda_minima = 0.3 * capacidade_maxima
    demanda_maxima = 0.9 * capacidade_maxima
    return np.clip(demanda_ajustada, demanda_minima, demanda_maxima)

def gerar_posicoes_ues_por_celula(id_celula, parametros_ambientais):
    posicoes_ues = []
    fatores_densidade = {
        'densa_urbana': (10, 20),
        'urbana': (5, 15),
        'suburbana': (3, 10),
        'rural': (2, 8)
    }
    
    min_ues, max_ues = fatores_densidade[parametros_ambientais.get('densidade_urbana', 'urbana')]
    num_ues = np.random.randint(min_ues, max_ues + 1)
    centro_x = 0
    centro_y = 0
    
    for _ in range(num_ues):
        r = RAIO_CELULA * np.sqrt(np.random.random())
        theta = np.random.uniform(0, 2*np.pi)
        x = centro_x + r * np.cos(theta)
        y = centro_y + r * np.sin(theta)
        h_BS = 10.0
        h_UT = 1.5
        posicoes_ues.append({
            'id_celula': id_celula,
            'x': x,
            'y': y,
            'distancia': r,
            'distancia_3d': np.sqrt(r**2 + (h_BS - h_UT)**2)
        })
    
    return posicoes_ues

def simular_iteracao_rede(iteracao, total_iteracoes):
    fatores_ambiente = obter_fatores_ambientais(iteracao, total_iteracoes)
    parametros_base = {}
    parametros_base = aplicar_efeitos_climaticos(fatores_ambiente['clima'], parametros_base)
    parametros_base = aplicar_efeitos_densidade_urbana(fatores_ambiente['densidade_urbana'], parametros_base)
    parametros_base = aplicar_efeitos_mobilidade(fatores_ambiente['mobilidade'], parametros_base)
    parametros_base.update(fatores_ambiente)
    
    todas_posicoes_ues = []
    for id_celula in range(NUM_CELULAS):
        ues_celula = gerar_posicoes_ues_por_celula(id_celula, fatores_ambiente)
        todas_posicoes_ues.extend(ues_celula)
    
    resultados_conexao = []
    for indice_ue, ue in enumerate(todas_posicoes_ues):
        id_celula = ue['id_celula']
        distancia_2d = ue['distancia']
        distancia_3d = ue['distancia_3d']
        prob_los = calcular_probabilidade_los(distancia_2d, fatores_ambiente['densidade_urbana'])
        condicao_los = 'LOS' if np.random.random() < prob_los else 'NLOS'
        perda_percurso = calcular_perda_percurso_3gpp(distancia_2d, distancia_3d, condicao_los, parametros_base)
        sombreamento = calcular_sombreamento(condicao_los, parametros_base)
        amostras_desvanecimento = gerar_amostras_desvanecimento_rapido(20, parametros_base)
        perda_desvanecimento = np.mean(amostras_desvanecimento)
        potencia_recebida = POTENCIA_TRANSMISSAO - perda_percurso - sombreamento - perda_desvanecimento
        ruido_termico = -174
        largura_banda_hz = LARGURAS_BANDA[id_celula] * 1e6
        potencia_ruido = ruido_termico + 10 * np.log10(largura_banda_hz) + 7
        interferencia = parametros_base.get('nivel_interferencia', 5.0)
        interferencia += np.random.normal(0, 2)
        ruido_mais_interferencia = 10**(potencia_ruido/10) + 10**(interferencia/10)
        sinr = potencia_recebida - 10 * np.log10(ruido_mais_interferencia)
        sinr_linear = 10**(sinr/10)
        eficiencia_espectral = np.log2(1 + sinr_linear)
        capacidade_enlace = largura_banda_hz * eficiencia_espectral / 1e6
        trafego_background = gerar_trafego_background_dinamico(id_celula, iteracao, fatores_ambiente)
        capacidade_disponivel = max(0.1, capacidade_enlace - trafego_background)
        
        eventos = []
        if np.random.random() < PROBABILIDADES_EVENTOS['handover']:
            eventos.append('handover')
        if np.random.random() < PROBABILIDADES_EVENTOS['pico_interferencia']:
            interferencia += np.random.uniform(5, 15)
            eventos.append('pico_interferencia')
        if np.random.random() < PROBABILIDADES_EVENTOS['violacao_qos'] and capacidade_disponivel < 5:
            eventos.append('violacao_qos')
        
        resultados_conexao.append({
            'id_celula': id_celula,
            'id_ue': indice_ue,
            'posicao': (ue['x'], ue['y']),
            'distancia': distancia_2d,
            'condicao_los': condicao_los,
            'probabilidade_los': prob_los,
            'perda_percurso_db': perda_percurso,
            'sombreamento_db': sombreamento,
            'desvanecimento_db': perda_desvanecimento,
            'potencia_recebida_db': potencia_recebida,
            'sinr_db': sinr,
            'capacidade_enlace_mbps': capacidade_enlace,
            'trafego_background_mbps': trafego_background,
            'capacidade_disponivel_mbps': capacidade_disponivel,
            'interferencia_db': interferencia,
            'eventos': eventos,
            'ambiente': fatores_ambiente.copy()
        })
    
    return resultados_conexao

def executar_simulacao_monte_carlo(num_iteracoes=50):
    todos_resultados = []
    historico_metricas = defaultdict(list)
    
    for iteracao in range(num_iteracoes):
        print(f"\nIteração {iteracao + 1}/{num_iteracoes}")
        print("=" * 50)
        resultados = simular_iteracao_rede(iteracao, num_iteracoes)
        todos_resultados.append(resultados)
        valores_sinr = [r['sinr_db'] for r in resultados]
        valores_capacidade = [r['capacidade_disponivel_mbps'] for r in resultados]
        valores_capacidade_total = [r['capacidade_enlace_mbps'] for r in resultados]
        contagem_los = sum(1 for r in resultados if r['condicao_los'] == 'LOS')
        historico_metricas['sinr_medio'].append(np.mean(valores_sinr))
        historico_metricas['capacidade_media'].append(np.mean(valores_capacidade))
        historico_metricas['capacidade_total_media'].append(np.mean(valores_capacidade_total))
        historico_metricas['taxa_los'].append(contagem_los / len(resultados) if resultados else 0)
        historico_metricas['violacoes_qos'].append(
            sum(1 for r in resultados if 'violacao_qos' in r['eventos'])
        )
        
        if resultados:
            ambiente = resultados[0]['ambiente']
            print(f"Ambiente: {ambiente['densidade_urbana']}, {ambiente['clima']}, {ambiente['periodo_dia']}")
            print(f"Mobilidade: {ambiente['mobilidade']}")
            print(f"UEs ativos: {len(resultados)}")
            print(f"SNR médio: {np.mean(valores_sinr):.1f} dB")
            print(f"Capacidade total média: {np.mean(valores_capacidade_total):.1f} Mbps")
            print(f"Capacidade disponível média: {np.mean(valores_capacidade):.1f} Mbps")
            print(f"Taxa LOS: {100*contagem_los/len(resultados):.1f}%")
    
    return todos_resultados, historico_metricas

def calcular_viabilidade_urllc(resultado):
    sinr_db = resultado['sinr_db']
    capacidade_disponivel = resultado['capacidade_disponivel_mbps']
    condicao_los = resultado['condicao_los']
    interferencia = resultado['interferencia_db']
    
    # Critérios realistas para URLLC
    # 1. SINR mínimo para confiabilidade
    # Para URLLC, precisamos de SINR mais alto
    sinr_minimo = 10 if condicao_los == 'LOS' else 15  # dB
    
    # 2. Capacidade mínima para suportar tráfego URLLC
    capacidade_minima_urllc = 2.0  # Mbps (para pacotes URLLC típicos)
    
    # 3. Estimativa de BER (Bit Error Rate) baseada no SINR
    if condicao_los == 'LOS':
        ber = 0.5 * np.exp(-10**(sinr_db/10) / 2)  # Aproximação para modulação BPSK
    else:
        ber = 0.5 / (1 + 10**(sinr_db/10))  # Rayleigh fading
    
    # 4. Confiabilidade estimada (1 - BER para um pacote de 100 bytes = 800 bits)
    confiabilidade_pacote = (1 - ber) ** 800
    
    # 5. Latência estimada (processamento + transmissão + fila)
    tamanho_pacote_bits = 800  # 100 bytes típicos para URLLC
    taxa_transmissao = capacidade_disponivel * 1e6  # bps
    if taxa_transmissao > 0:
        latencia_transmissao = (tamanho_pacote_bits / taxa_transmissao) * 1000  # ms
    else:
        latencia_transmissao = float('inf')
    
    latencia_processamento = 1.0  # ms
    latencia_fila = np.random.exponential(0.5)  # ms
    latencia_total = latencia_processamento + latencia_transmissao + latencia_fila
    
    # Critérios de viabilidade URLLC
    requisitos_atendidos = 0
    total_requisitos = 5
    
    if sinr_db >= sinr_minimo:
        requisitos_atendidos += 1
    
    if capacidade_disponivel >= capacidade_minima_urllc:
        requisitos_atendidos += 1
    
    if ber <= 1e-6:  # BER máximo para URLLC
        requisitos_atendidos += 1
    
    if confiabilidade_pacote >= 0.99999:  # 99.999% de confiabilidade
        requisitos_atendidos += 1
    
    if latencia_total <= 10:  # 10 ms máximo para URLLC
        requisitos_atendidos += 1
    
    # Viabilidade se atender pelo menos 4 dos 5 critérios
    viavel = requisitos_atendidos >= 4
    
    return {
        'viavel': viavel,
        'sinr_minimo_atendido': sinr_db >= sinr_minimo,
        'capacidade_minima_atendida': capacidade_disponivel >= capacidade_minima_urllc,
        'ber_atendido': ber <= 1e-6,
        'confiabilidade_atendida': confiabilidade_pacote >= 0.99999,
        'latencia_atendida': latencia_total <= 10,
        'ber_real': ber,
        'confiabilidade_real': confiabilidade_pacote,
        'latencia_total_ms': latencia_total,
        'sinr_minimo_requerido': sinr_minimo,
        'capacidade_minima_requerida': capacidade_minima_urllc,
        'requisitos_atendidos': requisitos_atendidos
    }

def analisar_resultados_simulacao(todos_resultados, historico_metricas):
    print("\n" + "="*60)
    print("ANÁLISE FINAL DA SIMULAÇÃO")
    print("="*60)
    resultados_planos = []
    for iteracao in todos_resultados:
        resultados_planos.extend(iteracao)
    
    print(f"\nTotal de amostras coletadas: {len(resultados_planos)}")
    print(f"Número de iterações: {len(todos_resultados)}")
    
    if not resultados_planos:
        print("Nenhum resultado para analisar!")
        return
    
    print("\nMétricas por célula:")
    print("-"*40)
    for id_celula in range(NUM_CELULAS):
        resultados_celula = [r for r in resultados_planos if r['id_celula'] == id_celula]
        if not resultados_celula:
            print(f"\nCélula {id_celula}: Nenhum dado disponível")
            continue
            
        valores_sinr = [r['sinr_db'] for r in resultados_celula]
        valores_capacidade = [r['capacidade_disponivel_mbps'] for r in resultados_celula]
        valores_capacidade_total = [r['capacidade_enlace_mbps'] for r in resultados_celula]
        print(f"\nCélula {id_celula} ({LARGURAS_BANDA[id_celula]} MHz):")
        print(f"  UEs simulados: {len(resultados_celula)}")
        print(f"  SNR médio: {np.mean(valores_sinr):.1f} ± {np.std(valores_sinr):.1f} dB")
        print(f"  Capacidade total média: {np.mean(valores_capacidade_total):.1f} Mbps")
        print(f"  Capacidade disponível média: {np.mean(valores_capacidade):.1f} Mbps")
        if len(resultados_celula) > 0:
            taxa_los = 100 * sum(1 for r in resultados_celula if r['condicao_los']=='LOS')/len(resultados_celula)
            print(f"  Taxa LOS: {taxa_los:.1f}%")
    
    print("\nImpacto das condições climáticas:")
    print("-"*40)
    for clima in VARIAVEIS_AMBIENTE['condicoes_climaticas']:
        resultados_clima = [r for r in resultados_planos if r['ambiente']['clima'] == clima]
        if resultados_clima:
            sinr_medio = np.mean([r['sinr_db'] for r in resultados_clima])
            capacidade_media = np.mean([r['capacidade_disponivel_mbps'] for r in resultados_clima])
            print(f"{clima.replace('_', ' ').title():15s}: SNR = {sinr_medio:.1f} dB, Capacidade = {capacidade_media:.1f} Mbps")
    
    plotar_resultados_simulacao(resultados_planos, historico_metricas)
    
    return resultados_planos

def plotar_resultados_simulacao(resultados_planos, historico_metricas):
    if not resultados_planos:
        print("Sem dados para plotar!")
        return
    
    fig, eixos = plt.subplots(2, 3, figsize=(15, 10))
    
    valores_sinr = [r['sinr_db'] for r in resultados_planos]
    eixos[0,0].hist(valores_sinr, bins=30, edgecolor='black', alpha=0.7)
    eixos[0,0].axvline(np.mean(valores_sinr), color='red', linestyle='--', 
                     label=f'Média: {np.mean(valores_sinr):.1f} dB')
    eixos[0,0].set_xlabel('SNR (dB)')
    eixos[0,0].set_ylabel('Frequência')
    eixos[0,0].set_title('Distribuição de SNR')
    eixos[0,0].legend()
    eixos[0,0].grid(True, alpha=0.3)
    
    distancias = [r['distancia'] for r in resultados_planos]
    eixos[0,1].scatter(distancias, valores_sinr, alpha=0.5, s=10)
    eixos[0,1].set_xlabel('Distância (m)')
    eixos[0,1].set_ylabel('SNR (dB)')
    eixos[0,1].set_title('SNR vs Distância')
    eixos[0,1].grid(True, alpha=0.3)
    
    dados_celula = defaultdict(list)
    for r in resultados_planos:
        dados_celula[r['id_celula']].append(r['capacidade_disponivel_mbps'])
    
    ids_celulas = list(dados_celula.keys())
    capacidades = [np.mean(dados_celula[cid]) for cid in ids_celulas]
    erros = [np.std(dados_celula[cid]) for cid in ids_celulas]
    pos_x = np.arange(len(ids_celulas))
    eixos[0,2].bar(pos_x, capacidades, yerr=erros, capsize=5, alpha=0.7)
    eixos[0,2].set_xticks(pos_x)
    eixos[0,2].set_xticklabels([f'Célula {cid}' for cid in ids_celulas])
    eixos[0,2].set_ylabel('Capacidade Média (Mbps)')
    eixos[0,2].set_title('Capacidade Disponível por Célula')
    eixos[0,2].grid(True, alpha=0.3)
    
    iteracoes = range(len(historico_metricas['sinr_medio']))
    if iteracoes:
        eixos[1,0].plot(iteracoes, historico_metricas['sinr_medio'], 'b-', linewidth=2)
        if len(historico_metricas['sinr_medio']) > 1:
            desvio_sinr = np.std(historico_metricas['sinr_medio'])
            eixos[1,0].fill_between(iteracoes, 
                                  np.array(historico_metricas['sinr_medio']) - desvio_sinr,
                                  np.array(historico_metricas['sinr_medio']) + desvio_sinr,
                                  alpha=0.2)
        eixos[1,0].set_xlabel('Iteração')
        eixos[1,0].set_ylabel('SNR Médio (dB)')
        eixos[1,0].set_title('Evolução do SNR ao Longo do Tempo')
        eixos[1,0].grid(True, alpha=0.3)
    
    if distancias:
        bins_distancia = np.linspace(0, max(distancias), 11)
        taxas_los = []
        centros_bins = []
        for i in range(len(bins_distancia)-1):
            bin_min, bin_max = bins_distancia[i], bins_distancia[i+1]
            resultados_bin = [r for r in resultados_planos if bin_min <= r['distancia'] < bin_max]
            if resultados_bin and len(resultados_bin) > 5:
                contagem_los = sum(1 for r in resultados_bin if r['condicao_los'] == 'LOS')
                taxas_los.append(contagem_los / len(resultados_bin))
                centros_bins.append((bin_min + bin_max) / 2)
        
        if centros_bins:
            eixos[1,1].plot(centros_bins, taxas_los, 'bo-', linewidth=2, markersize=6)
            eixos[1,1].set_xlabel('Distância (m)')
            eixos[1,1].set_ylabel('Probabilidade de LOS')
            eixos[1,1].set_title('Probabilidade de LOS vs Distância')
            eixos[1,1].grid(True, alpha=0.3)
    
    sinr_clima = {}
    for r in resultados_planos:
        clima = r['ambiente']['clima']
        if clima not in sinr_clima:
            sinr_clima[clima] = []
        sinr_clima[clima].append(r['sinr_db'])
    
    if sinr_clima:
        rotulos_clima = list(sinr_clima.keys())
        medias_clima = [np.mean(sinr_clima[w]) for w in rotulos_clima]
        desvios_clima = [np.std(sinr_clima[w]) for w in rotulos_clima]
        pos_x = np.arange(len(rotulos_clima))
        eixos[1,2].bar(pos_x, medias_clima, yerr=desvios_clima, capsize=5, alpha=0.7)
        eixos[1,2].set_xticks(pos_x)
        eixos[1,2].set_xticklabels([w.replace('_', ' ').title() for w in rotulos_clima], rotation=45)
        eixos[1,2].set_ylabel('SNR Médio (dB)')
        eixos[1,2].set_title('Impacto das Condições Climáticas no SNR')
        eixos[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("="*60)
    print("SIMULAÇÃO DE REDE 5G URLLC - MODELO PROBABILÍSTICO")
    print("="*60)
    print(f"Configuração: {NUM_CELULAS} células")
    print(f"Bandas: {LARGURAS_BANDA} MHz")
    print(f"Frequência: {FREQUENCIA_PORTADORA} GHz (n78)")
    print(f"Raio da célula: {RAIO_CELULA} m")
    print(f"Potência de transmissão: {POTENCIA_TRANSMISSAO} dBm")
    print()
    
    num_iteracoes = 10
    print(f"Executando {num_iteracoes} iterações Monte Carlo...")
    todos_resultados, historico_metricas = executar_simulacao_monte_carlo(num_iteracoes)
    resultados_planos = analisar_resultados_simulacao(todos_resultados, historico_metricas)
    
    if resultados_planos:
        print("\n" + "="*60)
        print("ANÁLISE DETALHADA DE VIABILIDADE URLLC")
        print("="*60)
        
        resultados_viabilidade = []
        for resultado in resultados_planos:
            viabilidade = calcular_viabilidade_urllc(resultado)
            resultados_viabilidade.append({
                **viabilidade,
                'id_celula': resultado['id_celula'],
                'distancia': resultado['distancia'],
                'condicao_los': resultado['condicao_los'],
                'sinr_db': resultado['sinr_db'],
                'capacidade_disponivel': resultado['capacidade_disponivel_mbps']
            })
        
        viaveis = [r for r in resultados_viabilidade if r['viavel']]
        taxa_viabilidade = 100 * len(viaveis) / len(resultados_viabilidade)
        
        print(f"\nTotal de conexões analisadas: {len(resultados_viabilidade)}")
        print(f"Conexões viáveis para URLLC: {len(viaveis)}")
        print(f"Taxa de viabilidade URLLC: {taxa_viabilidade:.1f}%")
        
        print("\nDistribuição dos critérios atendidos:")
        print("-"*40)
        for i in range(6):
            count = sum(1 for r in resultados_viabilidade if r['requisitos_atendidos'] == i)
            percent = 100 * count / len(resultados_viabilidade)
            print(f"{i} critérios atendidos: {count} conexões ({percent:.1f}%)")
        
        print("\nEstatísticas dos critérios individuais:")
        print("-"*40)
        criterios = ['sinr_minimo_atendido', 'capacidade_minima_atendida', 
                    'ber_atendido', 'confiabilidade_atendida', 'latencia_atendida']
        nomes = ['SNR mínimo', 'Capacidade mínima', 'BER ≤ 1e-6', 
                'Confiabilidade ≥ 99.999%', 'Latência ≤ 10ms']
        
        for criterio, nome in zip(criterios, nomes):
            count = sum(1 for r in resultados_viabilidade if r[criterio])
            percent = 100 * count / len(resultados_viabilidade)
            print(f"{nome}: {count} conexões ({percent:.1f}%)")
        
        print("\nAnálise por condição LOS/NLOS:")
        print("-"*40)
        for condicao in ['LOS', 'NLOS']:
            resultados_cond = [r for r in resultados_viabilidade if r['condicao_los'] == condicao]
            if resultados_cond:
                viaveis_cond = [r for r in resultados_cond if r['viavel']]
                taxa = 100 * len(viaveis_cond) / len(resultados_cond)
                sinr_medio = np.mean([r['sinr_db'] for r in resultados_cond])
                capacidade_media = np.mean([r['capacidade_disponivel'] for r in resultados_cond])
                print(f"\n{condicao}:")
                print(f"  Conexões: {len(resultados_cond)}")
                print(f"  Viabilidade: {taxa:.1f}%")
                print(f"  SNR médio: {sinr_medio:.1f} dB")
                print(f"  Capacidade média: {capacidade_media:.1f} Mbps")
        
        print("\nViabilidade URLLC por célula:")
        print("-"*40)
        for id_celula in range(NUM_CELULAS):
            resultados_cel = [r for r in resultados_viabilidade if r['id_celula'] == id_celula]
            if resultados_cel:
                viaveis_cel = [r for r in resultados_cel if r['viavel']]
                taxa = 100 * len(viaveis_cel) / len(resultados_cel)
                sinr_medio = np.mean([r['sinr_db'] for r in resultados_cel])
                capacidade_media = np.mean([r['capacidade_disponivel'] for r in resultados_cel])
                print(f"Célula {id_celula} ({LARGURAS_BANDA[id_celula]} MHz):")
                print(f"  Viabilidade: {taxa:.1f}%")
                print(f"  SNR médio: {sinr_medio:.1f} dB")
                print(f"  Capacidade média: {capacidade_media:.1f} Mbps")
        
        print("\nExemplos de conexões viáveis e não viáveis:")
        print("-"*40)
        print("\nTop 3 conexões mais viáveis:")
        mais_viaveis = sorted(resultados_viabilidade, 
                            key=lambda x: x['requisitos_atendidos'], reverse=True)[:3]
        for i, conexao in enumerate(mais_viaveis, 1):
            print(f"\n{i}. Célula {conexao['id_celula']}, {conexao['condicao_los']}, "
                  f"Distância: {conexao['distancia']:.1f}m")
            print(f"   SNR: {conexao['sinr_db']:.1f} dB, "
                  f"Capacidade: {conexao['capacidade_disponivel']:.1f} Mbps")
            print(f"   Critérios atendidos: {conexao['requisitos_atendidos']}/5")
            print(f"   BER: {conexao['ber_real']:.2e}, "
                  f"Confiabilidade: {conexao['confiabilidade_real']:.6f}, "
                  f"Latência: {conexao['latencia_total_ms']:.2f} ms")
        
        print("\nPrincipais causas de inviabilidade:")
        print("-"*40)
        nao_viaveis = [r for r in resultados_viabilidade if not r['viavel']]
        if nao_viaveis:
            causas = defaultdict(int)
            for conexao in nao_viaveis:
                if not conexao['sinr_minimo_atendido']:
                    causas['SNR baixo'] += 1
                if not conexao['capacidade_minima_atendida']:
                    causas['Capacidade insuficiente'] += 1
                if not conexao['ber_atendido']:
                    causas['BER alto'] += 1
                if not conexao['confiabilidade_atendida']:
                    causas['Confiabilidade baixa'] += 1
                if not conexao['latencia_atendida']:
                    causas['Latência alta'] += 1
            
            print("Causas mais frequentes:")
            for causa, count in sorted(causas.items(), key=lambda x: x[1], reverse=True)[:5]:
                percent = 100 * count / len(nao_viaveis)
                print(f"  {causa}: {count} conexões ({percent:.1f}%)")

if __name__ == "__main__":
    main()
