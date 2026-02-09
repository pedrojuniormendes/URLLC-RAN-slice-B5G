import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configurações da rede
NUM_CELLS = 5
CELL_BANDWIDTHS = [10, 10, 15, 15, 20]  # MHz
CARRIER_FREQ = 3.55  # GHz (banda n78)
CELL_RADIUS = 300  # metros
TX_POWER = 33  # dBm

# Configurações do ambiente (variáveis)
ENV_VARIABLES = {
    'weather_conditions': ['clear', 'rain', 'fog', 'storm'],
    'urban_density': ['dense_urban', 'urban', 'suburban', 'rural'],
    'time_of_day': ['peak', 'off_peak', 'night'],
    'mobility_patterns': ['static', 'pedestrian', 'vehicular']
}

# Probabilidades base para eventos
EVENT_PROBABILITIES = {
    'handover': 0.1,
    'interference_spike': 0.05,
    'channel_variation': 0.3,
    'qos_violation': 0.15
}

def get_environmental_factors(iteration, total_iterations):
    """Gera fatores ambientais que variam ao longo da simulação"""
    # Variação cíclica baseada na iteração
    time_factor = (iteration % 24) / 24  # Ciclo de 24 horas
    
    # Condições climáticas probabilísticas
    weather_probs = [0.6, 0.2, 0.15, 0.05]  # clear, rain, fog, storm
    weather = np.random.choice(ENV_VARIABLES['weather_conditions'], p=weather_probs)
    
    # Densidade urbana baseada no tempo
    if time_factor < 0.25 or time_factor > 0.75:  # Madrugada/noite
        urban_density = 'rural'
    elif 0.4 < time_factor < 0.6:  # Horário de pico
        urban_density = 'dense_urban'
    else:
        urban_density = np.random.choice(['urban', 'suburban'], p=[0.6, 0.4])
    
    # Hora do dia
    if time_factor < 0.25:
        time_of_day = 'night'
    elif 0.25 <= time_factor < 0.5:
        time_of_day = 'off_peak'
    else:
        time_of_day = 'peak'
    
    # Padrão de mobilidade
    if urban_density == 'dense_urban':
        mobility = np.random.choice(['pedestrian', 'vehicular'], p=[0.7, 0.3])
    else:
        mobility = np.random.choice(['static', 'pedestrian', 'vehicular'], p=[0.3, 0.4, 0.3])
    
    return {
        'weather': weather,
        'urban_density': urban_density,
        'time_of_day': time_of_day,
        'mobility': mobility,
        'time_factor': time_factor,
        'iteration': iteration
    }

def apply_weather_effects(weather, base_params):
    """Aplica efeitos das condições climáticas"""
    weather_effects = {
        'clear': {'pathloss_offset': 0, 'shadowing_std': 3.0, 'availability_factor': 1.0},
        'rain': {'pathloss_offset': 2.5, 'shadowing_std': 4.0, 'availability_factor': 0.95},
        'fog': {'pathloss_offset': 1.5, 'shadowing_std': 3.5, 'availability_factor': 0.98},
        'storm': {'pathloss_offset': 5.0, 'shadowing_std': 6.0, 'availability_factor': 0.85}
    }
    
    effects = weather_effects[weather].copy()
    
    # Adiciona ruído aleatório aos efeitos
    for key in effects:
        if key != 'availability_factor':
            effects[key] += np.random.normal(0, effects[key] * 0.1)
    
    return {**base_params, **effects}

def apply_urban_density_effects(density, base_params):
    """Aplica efeitos da densidade urbana"""
    density_effects = {
        'dense_urban': {
            'los_probability': 0.3,
            'interference_level': 8.0,
            'building_loss_mean': 20.0
        },
        'urban': {
            'los_probability': 0.5,
            'interference_level': 5.0,
            'building_loss_mean': 15.0
        },
        'suburban': {
            'los_probability': 0.7,
            'interference_level': 2.0,
            'building_loss_mean': 10.0
        },
        'rural': {
            'los_probability': 0.9,
            'interference_level': 1.0,
            'building_loss_mean': 5.0
        }
    }
    
    effects = density_effects[density].copy()
    
    # Variação estocástica
    for key in effects:
        effects[key] *= np.random.uniform(0.9, 1.1)
    
    return {**base_params, **effects}

def apply_mobility_effects(mobility, base_params, ue_velocity=None):
    """Aplica efeitos da mobilidade"""
    if ue_velocity is None:
        mobility_speeds = {
            'static': (0, 0.5),
            'pedestrian': (0.5, 5),
            'vehicular': (5, 30)
        }
        min_speed, max_speed = mobility_speeds[mobility]
        ue_velocity = np.random.uniform(min_speed, max_speed)
    
    mobility_effects = {
        'doppler_spread': ue_velocity * CARRIER_FREQ * 1e9 / 3e8,
        'handover_probability': min(0.05 + ue_velocity/100, 0.3),
        'channel_coherence_time': 0.423 / (ue_velocity * CARRIER_FREQ / 3e8) if ue_velocity > 0 else float('inf')
    }
    
    return {**base_params, **mobility_effects, 'ue_velocity': ue_velocity}

def calculate_los_probability(distance_2d, urban_density):
    """Probabilidade de LOS baseada em 3GPP com variações ambientais"""
    base_probabilities = {
        'dense_urban': 0.3,
        'urban': 0.5,
        'suburban': 0.7,
        'rural': 0.9
    }
    
    base_prob = base_probabilities[urban_density]
    
    # Decaimento com a distância
    if distance_2d <= 18:
        distance_factor = 1.0
    else:
        distance_factor = 18 / distance_2d
    
    # Variação aleatória
    random_variation = np.random.uniform(0.9, 1.1)
    
    return min(base_prob * distance_factor * random_variation, 1.0)

def calculate_pathloss_3gpp(distance_2d, distance_3d, los_condition, environmental_params):
    """Modelo de pathloss 3GPP com variações ambientais"""
    fc = CARRIER_FREQ
    h_BS = 10.0  # altura da BS
    h_UT = 1.5   # altura do UE
    
    # Efeitos ambientais
    weather_offset = environmental_params.get('pathloss_offset', 0)
    building_loss = np.random.normal(
        environmental_params.get('building_loss_mean', 10),
        environmental_params.get('building_loss_std', 3)
    )
    
    # Pathloss base
    if los_condition == 'LOS':
        if distance_2d <= 5 * h_BS * h_UT * fc / 3:
            pl = 21 * np.log10(distance_3d) + 20 * np.log10(fc) + 32.9
        else:
            pl = 40 * np.log10(distance_3d) + 20 * np.log10(fc) - 9.5
    else:  # NLOS
        pl_los = 21 * np.log10(distance_3d) + 20 * np.log10(fc) + 32.9
        pl_nlos = 35.3 * np.log10(distance_3d) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (h_UT - 1.5)
        pl = max(pl_los, pl_nlos)
    
    # Aplica variações ambientais
    pl_total = pl + weather_offset + building_loss
    
    # Adiciona componente aleatória
    pl_total += np.random.normal(0, environmental_params.get('pathloss_variation', 1.5))
    
    return max(pl_total, 60)  # Pathloss mínimo de 60 dB

def calculate_shadow_fading(los_condition, environmental_params):
    """Desvanecimento por sombreamento com variações"""
    base_sigma = 3.0 if los_condition == 'LOS' else 4.0
    
    # Aumenta variabilidade baseada na densidade urbana
    urban_factor = {
        'dense_urban': 1.5,
        'urban': 1.2,
        'suburban': 1.0,
        'rural': 0.8
    }[environmental_params.get('urban_density', 'urban')]
    
    sigma = base_sigma * urban_factor * np.random.uniform(0.9, 1.1)
    
    # Efeito do clima
    if environmental_params.get('weather') in ['rain', 'storm']:
        sigma *= 1.3
    
    return np.random.normal(0, sigma)

def generate_fast_fading_samples(num_samples, environmental_params):
    """Gera amostras de desvanecimento rápido - VERSÃO CORRIGIDA"""
    velocity = environmental_params.get('ue_velocity', 1.0)
    doppler_freq = velocity * CARRIER_FREQ * 1e9 / 3e8
    
    # Fator K (Rician) baseado em LOS probability
    los_prob = environmental_params.get('los_probability', 0.5)
    if np.random.random() < los_prob:
        k_db = np.random.uniform(7, 13)  # dB
    else:
        k_db = -np.inf
    
    k_factor = 10 ** (k_db / 10) if k_db > -np.inf else 0
    
    # Componente LOS
    t = np.linspace(0, 1, num_samples)
    
    if k_factor > 0:
        los_phase = 2 * np.pi * doppler_freq * t + np.random.uniform(0, 2*np.pi)
        los_component = np.sqrt(k_factor/(k_factor + 1)) * np.exp(1j * los_phase)
    else:
        los_component = 0
    
    # Componente NLOS (Rayleigh) - VERSÃO SIMPLIFICADA E CORRETA
    # Gera ruído complexo Gaussiano
    nlos_complex = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
    
    # Aplica efeito Doppler se necessário
    if doppler_freq > 0 and num_samples > 1:
        # Filtro passa-baixa para simular correlação temporal
        correlation_time = min(0.1 / doppler_freq, 0.5)
        alpha = np.exp(-1 / (correlation_time * num_samples))
        
        # Filtro IIR simples para correlação temporal
        nlos_filtered = np.zeros_like(nlos_complex, dtype=complex)
        nlos_filtered[0] = nlos_complex[0]
        for i in range(1, num_samples):
            nlos_filtered[i] = alpha * nlos_filtered[i-1] + np.sqrt(1-alpha**2) * nlos_complex[i]
        
        nlos_component = np.sqrt(1/(k_factor + 1)) * nlos_filtered
    else:
        nlos_component = np.sqrt(1/(k_factor + 1)) * nlos_complex
    
    # Canal combinado
    channel = los_component + nlos_component
    
    # Retorna magnitude em dB
    return 20 * np.log10(np.abs(channel) + 1e-10)  # +1e-10 para evitar log(0)

def generate_dynamic_background_traffic(cell_id, iteration, environmental_params):
    """Gera tráfego de background com variações temporais e espaciais"""
    base_capacity = {
        10: 50,   # Mbps para 10 MHz
        15: 75,   # Mbps para 15 MHz
        20: 100   # Mbps para 20 MHz
    }
    
    bw = CELL_BANDWIDTHS[cell_id]
    max_capacity = base_capacity[bw]
    
    # Variação baseada no horário
    time_factor = environmental_params.get('time_factor', 0.5)
    time_multiplier = 0.3 + 0.7 * (1 - np.cos(2 * np.pi * time_factor)) / 2
    
    # Variação baseada na densidade urbana
    density_multiplier = {
        'dense_urban': 1.3,
        'urban': 1.1,
        'suburban': 0.9,
        'rural': 0.7
    }[environmental_params.get('urban_density', 'urban')]
    
    # Demanda base
    base_demand = np.random.uniform(0.5, 0.75) * max_capacity
    
    # Aplica variações
    adjusted_demand = base_demand * time_multiplier * density_multiplier
    
    # Adiciona flutuações aleatórias
    random_fluctuation = np.random.normal(1, 0.1)
    adjusted_demand *= random_fluctuation
    
    # Limita entre 30-90% da capacidade
    min_demand = 0.3 * max_capacity
    max_demand = 0.9 * max_capacity
    
    return np.clip(adjusted_demand, min_demand, max_demand)

def generate_urllc_traffic_pattern(iteration, environmental_params):
    """Gera padrão de tráfego URLLC com burstiness"""
    time_factor = environmental_params.get('time_factor', 0.5)
    
    # Tráfego mais intenso em horários de pico
    if environmental_params.get('time_of_day') == 'peak':
        arrival_rate = np.random.poisson(5)  # Alta taxa
        burst_probability = 0.3
    else:
        arrival_rate = np.random.poisson(2)  # Baixa taxa
        burst_probability = 0.1
    
    # Verifica se ocorre burst de tráfego
    has_burst = np.random.random() < burst_probability
    
    if has_burst:
        # Burst de tráfego URLLC
        num_requests = np.random.randint(5, 15)
        burst_duration = np.random.uniform(0.1, 1.0)  # segundos
    else:
        num_requests = max(1, np.random.poisson(arrival_rate))
        burst_duration = 0
    
    # Requisitos de QoS variáveis
    qos_requirements = []
    for _ in range(num_requests):
        latency_req = np.random.uniform(1, 10)  # ms
        reliability_req = np.random.uniform(0.999, 0.99999)
        bw_req = np.random.uniform(0.5, 5)  # Mbps
        
        qos_requirements.append({
            'latency_ms': latency_req,
            'reliability': reliability_req,
            'bandwidth_mbps': bw_req,
            'priority': np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.5, 0.3])
        })
    
    return {
        'num_requests': num_requests,
        'qos_requirements': qos_requirements,
        'has_burst': has_burst,
        'burst_duration': burst_duration,
        'time_factor': time_factor
    }

def generate_ue_positions_per_cell(cell_id, environmental_params):
    """Gera posições dos UEs para uma célula específica"""
    ue_positions = []
    
    # Determina número de UEs baseado na densidade urbana
    density_factors = {
        'dense_urban': (10, 20),
        'urban': (5, 15),
        'suburban': (3, 10),
        'rural': (2, 8)
    }
    
    min_ues, max_ues = density_factors[environmental_params.get('urban_density', 'urban')]
    num_ues = np.random.randint(min_ues, max_ues + 1)
    
    # Centro da célula (simplificado - todas as células no mesmo ponto para esta demo)
    cell_center_x = 0
    cell_center_y = 0
    
    for _ in range(num_ues):
        # Distribuição não-uniforme (mais UEs próximos à BS)
        r = CELL_RADIUS * np.sqrt(np.random.random())  # Distribuição triangular
        theta = np.random.uniform(0, 2*np.pi)
        x = cell_center_x + r * np.cos(theta)
        y = cell_center_y + r * np.sin(theta)
        
        # Alturas da BS e UE
        h_BS = 10.0  # metros
        h_UT = 1.5   # metros
        
        ue_positions.append({
            'cell_id': cell_id,
            'x': x,
            'y': y,
            'distance': r,
            'distance_3d': np.sqrt(r**2 + (h_BS - h_UT)**2)
        })
    
    return ue_positions

def simulate_network_iteration(iteration, total_iterations):
    """Executa uma iteração completa da simulação"""
    # 1. Obtém condições ambientais
    env_factors = get_environmental_factors(iteration, total_iterations)
    
    # 2. Aplica efeitos ambientais aos parâmetros
    base_params = {}
    base_params = apply_weather_effects(env_factors['weather'], base_params)
    base_params = apply_urban_density_effects(env_factors['urban_density'], base_params)
    base_params = apply_mobility_effects(env_factors['mobility'], base_params)
    base_params.update(env_factors)
    
    # 3. Gera posições dos UEs para todas as células
    all_ue_positions = []
    for cell_id in range(NUM_CELLS):
        cell_ues = generate_ue_positions_per_cell(cell_id, env_factors)
        all_ue_positions.extend(cell_ues)
    
    # 4. Simula cada conexão UE-BS
    connection_results = []
    
    for ue_idx, ue in enumerate(all_ue_positions):
        cell_id = ue['cell_id']
        distance_2d = ue['distance']
        distance_3d = ue['distance_3d']
        
        # Determina condição LOS/NLOS probabilisticamente
        los_prob = calculate_los_probability(distance_2d, env_factors['urban_density'])
        los_condition = 'LOS' if np.random.random() < los_prob else 'NLOS'
        
        # Calcula pathloss
        pathloss = calculate_pathloss_3gpp(distance_2d, distance_3d, los_condition, base_params)
        
        # Calcula shadow fading
        shadow_fading = calculate_shadow_fading(los_condition, base_params)
        
        # Calcula fast fading (média de várias amostras)
        fast_fading_samples = generate_fast_fading_samples(20, base_params)
        fast_fading_loss = np.mean(fast_fading_samples)
        
        # Potência recebida
        rx_power = TX_POWER - pathloss - shadow_fading - fast_fading_loss
        
        # Ruído e interferência
        thermal_noise = -174  # dBm/Hz
        bandwidth_hz = CELL_BANDWIDTHS[cell_id] * 1e6
        noise_power = thermal_noise + 10 * np.log10(bandwidth_hz) + 7  # NF = 7 dB
        
        # Interferência baseada no ambiente
        interference = base_params.get('interference_level', 5.0)
        interference += np.random.normal(0, 2)  # Variação aleatória
        
        # SINR
        noise_plus_interference = 10**(noise_power/10) + 10**(interference/10)
        sinr = rx_power - 10 * np.log10(noise_plus_interference)
        
        # Capacidade do link (Shannon simplificado)
        sinr_linear = 10**(sinr/10)
        spectral_efficiency = np.log2(1 + sinr_linear)
        link_capacity = bandwidth_hz * spectral_efficiency / 1e6  # Mbps
        
        # Tráfego de background
        background_traffic = generate_dynamic_background_traffic(cell_id, iteration, env_factors)
        
        # Capacidade disponível para URLLC
        available_capacity = max(0.1, link_capacity - background_traffic)  # mínimo 0.1 Mbps
        
        # Verifica eventos probabilísticos
        events = []
        if np.random.random() < EVENT_PROBABILITIES['handover']:
            events.append('handover')
        if np.random.random() < EVENT_PROBABILITIES['interference_spike']:
            interference += np.random.uniform(5, 15)
            events.append('interference_spike')
        if np.random.random() < EVENT_PROBABILITIES['qos_violation'] and available_capacity < 5:
            events.append('qos_violation')
        
        connection_results.append({
            'cell_id': cell_id,
            'ue_id': ue_idx,
            'position': (ue['x'], ue['y']),
            'distance': distance_2d,
            'los_condition': los_condition,
            'los_probability': los_prob,
            'pathloss_db': pathloss,
            'shadow_fading_db': shadow_fading,
            'fast_fading_db': fast_fading_loss,
            'rx_power_db': rx_power,
            'sinr_db': sinr,
            'link_capacity_mbps': link_capacity,
            'background_traffic_mbps': background_traffic,
            'available_capacity_mbps': available_capacity,
            'interference_db': interference,
            'events': events,
            'environment': env_factors.copy()  # Usar cópia para evitar referências
        })
    
    return connection_results

def run_monte_carlo_simulation(num_iterations=50):
    """Executa simulação Monte Carlo com múltiplas iterações"""
    all_results = []
    metrics_history = defaultdict(list)
    
    for iteration in range(num_iterations):
        print(f"\nIteração {iteration + 1}/{num_iterations}")
        print("=" * 50)
        
        # Executa iteração
        results = simulate_network_iteration(iteration, num_iterations)
        all_results.append(results)
        
        # Calcula métricas agregadas
        sinr_values = [r['sinr_db'] for r in results]
        capacity_values = [r['available_capacity_mbps'] for r in results]
        los_count = sum(1 for r in results if r['los_condition'] == 'LOS')
        
        # Armazena métricas
        metrics_history['avg_sinr'].append(np.mean(sinr_values))
        metrics_history['avg_capacity'].append(np.mean(capacity_values))
        metrics_history['los_ratio'].append(los_count / len(results) if results else 0)
        metrics_history['qos_violations'].append(
            sum(1 for r in results if 'qos_violation' in r['events'])
        )
        
        # Exibe condições ambientais
        if results:
            env = results[0]['environment']
            print(f"Ambiente: {env['urban_density']}, {env['weather']}, {env['time_of_day']}")
            print(f"Mobilidade: {env['mobility']}")
            print(f"UEs ativos: {len(results)}")
            print(f"SNR médio: {np.mean(sinr_values):.1f} dB")
            print(f"Capacidade disponível média: {np.mean(capacity_values):.1f} Mbps")
            print(f"Taxa LOS: {100*los_count/len(results):.1f}%")
    
    return all_results, metrics_history

def analyze_simulation_results(all_results, metrics_history):
    """Analisa e visualiza os resultados da simulação"""
    # 1. Estatísticas gerais
    print("\n" + "="*60)
    print("ANÁLISE FINAL DA SIMULAÇÃO")
    print("="*60)
    
    # Achata todos os resultados
    flat_results = []
    for iteration in all_results:
        flat_results.extend(iteration)
    
    print(f"\nTotal de amostras coletadas: {len(flat_results)}")
    print(f"Número de iterações: {len(all_results)}")
    
    if not flat_results:
        print("Nenhum resultado para analisar!")
        return
    
    # 2. Métricas por célula
    print("\nMétricas por célula:")
    print("-"*40)
    
    for cell_id in range(NUM_CELLS):
        cell_results = [r for r in flat_results if r['cell_id'] == cell_id]
        if not cell_results:
            print(f"\nCélula {cell_id}: Nenhum dado disponível")
            continue
            
        sinr_vals = [r['sinr_db'] for r in cell_results]
        cap_vals = [r['available_capacity_mbps'] for r in cell_results]
        
        print(f"\nCélula {cell_id} ({CELL_BANDWIDTHS[cell_id]} MHz):")
        print(f"  UEs simulados: {len(cell_results)}")
        print(f"  SNR médio: {np.mean(sinr_vals):.1f} ± {np.std(sinr_vals):.1f} dB")
        print(f"  Capacidade média: {np.mean(cap_vals):.1f} ± {np.std(cap_vals):.1f} Mbps")
        if len(cell_results) > 0:
            los_ratio = 100 * sum(1 for r in cell_results if r['los_condition']=='LOS')/len(cell_results)
            print(f"  Taxa LOS: {los_ratio:.1f}%")
    
    # 3. Efeitos ambientais
    print("\nImpacto das condições ambientais:")
    print("-"*40)
    
    for weather in ENV_VARIABLES['weather_conditions']:
        weather_results = [r for r in flat_results if r['environment']['weather'] == weather]
        if weather_results:
            avg_sinr = np.mean([r['sinr_db'] for r in weather_results])
            print(f"{weather.capitalize():10s}: SNR médio = {avg_sinr:.1f} dB")
    
    # 4. Visualizações
    plot_simulation_results(flat_results, metrics_history)

def plot_simulation_results(flat_results, metrics_history):
    """Gera gráficos dos resultados"""
    if not flat_results:
        print("Sem dados para plotar!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribuição de SNR
    sinr_values = [r['sinr_db'] for r in flat_results]
    axes[0,0].hist(sinr_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0,0].axvline(np.mean(sinr_values), color='red', linestyle='--', 
                     label=f'Média: {np.mean(sinr_values):.1f} dB')
    axes[0,0].set_xlabel('SNR (dB)')
    axes[0,0].set_ylabel('Frequência')
    axes[0,0].set_title('Distribuição de SNR')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. SNR vs Distância
    distances = [r['distance'] for r in flat_results]
    axes[0,1].scatter(distances, sinr_values, alpha=0.5, s=10)
    axes[0,1].set_xlabel('Distância (m)')
    axes[0,1].set_ylabel('SNR (dB)')
    axes[0,1].set_title('SNR vs Distância')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Capacidade disponível por célula
    cell_data = defaultdict(list)
    for r in flat_results:
        cell_data[r['cell_id']].append(r['available_capacity_mbps'])
    
    cell_ids = list(cell_data.keys())
    capacities = [np.mean(cell_data[cid]) for cid in cell_ids]
    errors = [np.std(cell_data[cid]) for cid in cell_ids]
    
    x_pos = np.arange(len(cell_ids))
    axes[0,2].bar(x_pos, capacities, yerr=errors, capsize=5, alpha=0.7)
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels([f'Célula {cid}' for cid in cell_ids])
    axes[0,2].set_ylabel('Capacidade Média (Mbps)')
    axes[0,2].set_title('Capacidade Disponível por Célula')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Evolução temporal das métricas
    iterations = range(len(metrics_history['avg_sinr']))
    if iterations:
        axes[1,0].plot(iterations, metrics_history['avg_sinr'], 'b-', linewidth=2)
        if len(metrics_history['avg_sinr']) > 1:
            sinr_std = np.std(metrics_history['avg_sinr'])
            axes[1,0].fill_between(iterations, 
                                  np.array(metrics_history['avg_sinr']) - sinr_std,
                                  np.array(metrics_history['avg_sinr']) + sinr_std,
                                  alpha=0.2)
        axes[1,0].set_xlabel('Iteração')
        axes[1,0].set_ylabel('SNR Médio (dB)')
        axes[1,0].set_title('Evolução do SNR ao Longo do Tempo')
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Taxa LOS por distância
    if distances:
        distance_bins = np.linspace(0, max(distances), 11)
        los_ratios = []
        bin_centers = []
        
        for i in range(len(distance_bins)-1):
            bin_min, bin_max = distance_bins[i], distance_bins[i+1]
            bin_results = [r for r in flat_results if bin_min <= r['distance'] < bin_max]
            if bin_results and len(bin_results) > 5:  # Pelo menos 5 amostras
                los_count = sum(1 for r in bin_results if r['los_condition'] == 'LOS')
                los_ratios.append(los_count / len(bin_results))
                bin_centers.append((bin_min + bin_max) / 2)
        
        if bin_centers:
            axes[1,1].plot(bin_centers, los_ratios, 'bo-', linewidth=2, markersize=6)
            axes[1,1].set_xlabel('Distância (m)')
            axes[1,1].set_ylabel('Probabilidade de LOS')
            axes[1,1].set_title('Probabilidade de LOS vs Distância')
            axes[1,1].grid(True, alpha=0.3)
    
    # 6. Efeito das condições climáticas
    weather_sinr = {}
    for r in flat_results:
        weather = r['environment']['weather']
        if weather not in weather_sinr:
            weather_sinr[weather] = []
        weather_sinr[weather].append(r['sinr_db'])
    
    if weather_sinr:
        weather_labels = list(weather_sinr.keys())
        weather_means = [np.mean(weather_sinr[w]) for w in weather_labels]
        weather_stds = [np.std(weather_sinr[w]) for w in weather_labels]
        
        x_pos = np.arange(len(weather_labels))
        axes[1,2].bar(x_pos, weather_means, yerr=weather_stds, capsize=5, alpha=0.7)
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels([w.capitalize() for w in weather_labels], rotation=45)
        axes[1,2].set_ylabel('SNR Médio (dB)')
        axes[1,2].set_title('Impacto das Condições Climáticas no SNR')
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal de execução"""
    print("="*60)
    print("SIMULAÇÃO DE REDE 5G URLLC - MODELO PROBABILÍSTICO")
    print("="*60)
    print(f"Configuração: {NUM_CELLS} células")
    print(f"Bandas: {CELL_BANDWIDTHS} MHz")
    print(f"Frequência: {CARRIER_FREQ} GHz (n78)")
    print(f"Raio da célula: {CELL_RADIUS} m")
    print()
    
    # Configurar seed para reprodutibilidade (opcional)
    # np.random.seed(42)
    
    # Executa simulação Monte Carlo
    num_iterations = 20  # Reduzido para demonstração rápida
    print(f"Executando {num_iterations} iterações Monte Carlo...")
    
    all_results, metrics_history = run_monte_carlo_simulation(num_iterations)
    
    # Analisa resultados
    analyze_simulation_results(all_results, metrics_history)
    
    # Análise adicional
    if all_results and all_results[-1]:
        print("\n" + "="*60)
        print("ANÁLISE DE DISPONIBILIDADE URLLC")
        print("="*60)
        
        # Verifica viabilidade URLLC (latência < 10ms, reliability > 99.999%)
        urllc_feasible = []
        last_iteration = all_results[-1]
        
        for r in last_iteration:
            # Critérios simplificados para URLLC
            # Latência estimada = latência de processamento + latência de transmissão
            transmission_latency = 100 / max(r['available_capacity_mbps'], 0.1)  # ms para 100 bits
            total_latency = 1 + transmission_latency  # 1ms de processamento
            
            # Confiabilidade estimada baseada no SINR
            sinr_linear = 10**(r['sinr_db']/20)  # Aproximação
            reliability_estimate = 1 - np.exp(-sinr_linear)
            
            is_feasible = (total_latency < 10) and (reliability_estimate > 0.99999)
            urllc_feasible.append(is_feasible)
        
        if urllc_feasible:
            feasibility_rate = 100 * sum(urllc_feasible) / len(urllc_feasible)
            print(f"Taxa de viabilidade URLLC: {feasibility_rate:.1f}%")
            
            # Análise por tipo de ambiente
            print("\nViabilidade URLLC por ambiente:")
            print("-"*40)
            
            env_types = set()
            for r in last_iteration:
                env = r['environment']
                env_types.add((env['urban_density'], env['weather']))
            
            for density, weather in sorted(env_types):
                env_results = [r for r in last_iteration 
                              if r['environment']['urban_density'] == density 
                              and r['environment']['weather'] == weather]
                
                if env_results:
                    feasible_count = 0
                    for r in env_results:
                        transmission_latency = 100 / max(r['available_capacity_mbps'], 0.1)
                        total_latency = 1 + transmission_latency
                        sinr_linear = 10**(r['sinr_db']/20)
                        reliability_estimate = 1 - np.exp(-sinr_linear)
                        
                        if total_latency < 10 and reliability_estimate > 0.99999:
                            feasible_count += 1
                    
                    ratio = 100 * feasible_count / len(env_results)
                    print(f"{density.replace('_', ' ').title():15s} - {weather.capitalize():8s}: {ratio:5.1f}% viável")

if __name__ == "__main__":
    main()