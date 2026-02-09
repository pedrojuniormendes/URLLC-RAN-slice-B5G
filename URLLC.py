import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class NRChannelModel:
    """
    Modelo de canal 3GPP para banda n78 (3.5 GHz)
    Baseado em 3GPP TR 38.901
    """
    
    def __init__(self, 
                 carrier_freq: float = 3.55,  # GHz
                 bs_height: float = 10.0,     # metros
                 ut_height: float = 1.5,      # metros
                 scenario: str = 'UMi',       # UMi ou RMa
                 los_probability: float = 0.8):
        
        self.fc = carrier_freq  # GHz
        self.h_BS = bs_height   # altura da BS
        self.h_UT = ut_height   # altura do UE
        self.scenario = scenario
        self.los_prob = los_probability
        
        # Parâmetros do modelo
        self.params = {
            'UMi': {
                'LOS': {'alpha': 21, 'beta': 32.9, 'breakpoint': 5*self.h_BS*self.h_UT*self.fc/3},
                'NLOS': {'alpha': 35.3, 'beta': 22.4, 'gamma': 21.3}
            },
            'RMa': {
                'LOS': {'alpha': 20, 'beta': 46.3, 'breakpoint': 2*np.pi*self.h_BS*self.h_UT*self.fc/3},
                'NLOS': {'alpha': 40, 'beta': 30.8}
            }
        }
    
    def calculate_pathloss(self, 
                          distance_2d: float,
                          distance_3d: float = None,
                          los_condition: str = None) -> Tuple[float, str]:
        """
        Calcula o pathloss para uma dada distância
        """
        if distance_3d is None:
            distance_3d = np.sqrt(distance_2d**2 + (self.h_BS - self.h_UT)**2)
        
        # Determina condição LOS/NLOS
        if los_condition is None:
            # Probabilidade de LOS baseada na distância (3GPP 38.901)
            if self.scenario == 'UMi':
                if distance_2d <= 18:
                    p_los = 1.0
                else:
                    p_los = (18/distance_2d + np.exp(-distance_2d/36)*(1 - 18/distance_2d))
            else:  # RMa
                if distance_2d <= 10:
                    p_los = 1.0
                else:
                    p_los = np.exp(-(distance_2d - 10)/1000)
            
            los_condition = 'LOS' if np.random.random() < min(p_los, self.los_prob) else 'NLOS'
        
        # Cálculo do pathloss
        if self.scenario == 'UMi':
            if los_condition == 'LOS':
                # Para distâncias antes do breakpoint
                if distance_2d <= 5*self.h_BS*self.h_UT*self.fc/3:
                    pl = 21*np.log10(distance_3d) + 20*np.log10(self.fc) + 32.9
                else:
                    pl = 40*np.log10(distance_3d) + 20*np.log10(self.fc) - 9.5
            else:  # NLOS
                pl_los = 21*np.log10(distance_3d) + 20*np.log10(self.fc) + 32.9
                pl_nlos = 35.3*np.log10(distance_3d) + 22.4 + 21.3*np.log10(self.fc) - 0.3*(self.h_UT - 1.5)
                pl = max(pl_los, pl_nlos)
        
        else:  # RMa
            if los_condition == 'LOS':
                if distance_2d <= 10:
                    pl = 20*np.log10(40*np.pi*distance_3d*self.fc/3)
                else:
                    pl = 20*np.log10(40*np.pi*distance_3d*self.fc/3) + 0.03*distance_2d**1.72
            else:  # NLOS
                pl = 161.04 - 7.1*np.log10(10) + 7.5*np.log10(distance_2d) - (24.37 - 3.7*(self.h_BS/self.h_UT)**2)*np.log10(self.h_BS) + (43.42 - 3.1*np.log10(self.h_BS))*(np.log10(distance_3d) - 3) + 20*np.log10(self.fc) - (3.2*(np.log10(11.75*self.h_UT))**2 - 4.97)
        
        return pl, los_condition
    
    def calculate_shadow_fading(self, los_condition: str) -> float:
        """
        Desvanecimento por sombreamento (lognormal)
        """
        if los_condition == 'LOS':
            sigma_sf = 3.0  # dB para LOS
        else:
            sigma_sf = 4.0  # dB para NLOS
        
        return np.random.normal(0, sigma_sf)
    
    def calculate_fast_fading(self, 
                             num_samples: int = 1000,
                             velocity: float = 3.0) -> np.ndarray:
        """
        Modelo de desvanecimento rápido (Rayleigh/Rician)
        """
        # Fator K para Rician (K em dB)
        k_factor_db = 7 if np.random.random() < self.los_prob else -np.inf  # Rayleigh para NLOS
        k_factor = 10**(k_factor_db/10)
        
        # Componente LOS
        if k_factor_db > -np.inf:
            los_component = np.sqrt(k_factor/(k_factor + 1))
        else:
            los_component = 0
        
        # Componente NLOS (Rayleigh)
        nlos_component = np.sqrt(1/(k_factor + 1))
        
        # Simulação do canal
        t = np.linspace(0, 1, num_samples)
        fd = velocity * self.fc * 1e9 / 3e8  # Frequência Doppler
        
        # Componentes I e Q
        xi = np.random.randn(num_samples)
        xq = np.random.randn(num_samples)
        
        # Filtro Doppler
        t_arrival = np.arange(-10, 10, 0.1)
        h = np.sqrt(1/(np.pi*fd*np.sqrt(1 - (t_arrival/fd)**2)))
        h[np.isnan(h)] = 0
        
        # Convolução para efeito Doppler
        xi = np.convolve(xi, h, mode='same')
        xq = np.convolve(xq, h, mode='same')
        
        # Normalização
        xi = xi / np.std(xi)
        xq = xq / np.std(xq)
        
        # Sinal resultante
        signal = los_component + nlos_component * (xi + 1j*xq) / np.sqrt(2)
        
        return 20*np.log10(np.abs(signal))
    
    def calculate_snr(self, 
                     tx_power: float,  # dBm
                     distance: float,
                     bandwidth: float,  # MHz
                     noise_figure: float = 7.0) -> Dict:
        """
        Calcula SNR considerando todos os efeitos do canal
        """
        # Pathloss
        pl, los_condition = self.calculate_pathloss(distance)
        
        # Shadow fading
        sf = self.calculate_shadow_fading(los_condition)
        
        # Perda total
        total_loss = pl + sf
        
        # Potência recebida
        rx_power = tx_power - total_loss  # dBm
        
        # Potência do ruído
        thermal_noise = -174  # dBm/Hz
        noise_power = thermal_noise + 10*np.log10(bandwidth * 1e6) + noise_figure
        
        # SNR
        snr = rx_power - noise_power
        
        return {
            'snr_db': snr,
            'rx_power_db': rx_power,
            'pathloss_db': pl,
            'shadow_fading_db': sf,
            'los_condition': los_condition,
            'total_loss_db': total_loss
        }

class NetworkSimulator:
    """
    Simulador de rede 5G com múltiplas células
    """
    
    def __init__(self, 
                 n_cells: int = 5,
                 bandwidths: list = [10, 10, 15, 15, 20],  # MHz
                 cell_radius: float = 500.0,  # metros
                 tx_power: float = 30.0):     # dBm
        
        self.n_cells = n_cells
        self.bandwidths = bandwidths
        self.cell_radius = cell_radius
        self.tx_power = tx_power
        
        # Posicionamento das células (hexagonal simplificado)
        self.bs_positions = self._generate_bs_positions()
        
        # Modelos de canal para cada célula
        self.channel_models = [NRChannelModel() for _ in range(n_cells)]
        
        # Capacidade de cada célula (simplificado)
        self.cell_capacities = self._calculate_cell_capacities()
        
    def _generate_bs_positions(self) -> np.ndarray:
        """Gera posições das estações base em layout hexagonal"""
        positions = []
        for i in range(self.n_cells):
            if i == 0:
                positions.append([0, 0])  # Célula central
            else:
                angle = 2 * np.pi * (i-1) / (self.n_cells-1)
                radius = self.cell_radius * 1.5
                positions.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(positions)
    
    def _calculate_cell_capacities(self) -> np.ndarray:
        """Calcula capacidade teórica por célula"""
        capacities = []
        for bw in self.bandwidths:
            # Capacidade aproximada considerando eficiência espectral típica
            spectral_efficiency = 3.5  # bps/Hz (conservador para 5G)
            capacity = bw * 1e6 * spectral_efficiency / 1e6  # Mbps
            capacities.append(capacity)
        return np.array(capacities)
    
    def generate_background_traffic(self) -> np.ndarray:
        """
        Gera tráfego de background entre 50-75% da capacidade
        """
        background_demand = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            min_demand = 0.5 * self.cell_capacities[i]
            max_demand = 0.75 * self.cell_capacities[i]
            background_demand[i] = np.random.uniform(min_demand, max_demand)
        return background_demand
    
    def simulate_urllc_connection(self,
                                 cell_id: int,
                                 ue_distance: float,
                                 urllc_bw_requirement: float = 5.0) -> Dict:
        """
        Simula uma conexão URLLC
        """
        # Calcula SNR
        channel_info = self.channel_models[cell_id].calculate_snr(
            tx_power=self.tx_power,
            distance=ue_distance,
            bandwidth=self.bandwidths[cell_id]
        )
        
        # Calcula capacidade disponível para URLLC
        background = self.generate_background_traffic()
        available_capacity = self.cell_capacities[cell_id] - background[cell_id]
        
        # Probabilidade de erro baseada na SNR
        snr_linear = 10**(channel_info['snr_db']/10)
        if channel_info['los_condition'] == 'LOS':
            ber = 0.5 * np.exp(-snr_linear/2)  # BPSK approximation
        else:
            ber = 0.5 / (1 + snr_linear)  # Rayleigh fading
        
        # Latência aproximada (simplificada)
        processing_delay = 1.0  # ms
        transmission_delay = (urllc_bw_requirement * 8) / max(available_capacity, 0.1)  # ms
        
        # Efeito do fast fading
        fast_fading = self.channel_models[cell_id].calculate_fast_fading(num_samples=10)
        
        return {
            'cell_id': cell_id,
            'ue_distance': ue_distance,
            'snr_db': channel_info['snr_db'],
            'ber': ber,
            'available_capacity_mbps': max(available_capacity, 0),
            'urllc_bw_requirement_mbps': urllc_bw_requirement,
            'total_latency_ms': processing_delay + transmission_delay,
            'los_condition': channel_info['los_condition'],
            'success_probability': 1 - ber,
            'fast_fading_variation_db': np.std(fast_fading)
        }
    
    def plot_pathloss_curves(self):
        """Plota curvas de pathloss para diferentes condições"""
        distances = np.linspace(10, 1000, 100)
        pl_los = np.zeros_like(distances)
        pl_nlos = np.zeros_like(distances)
        
        for i, d in enumerate(distances):
            pl_los[i], _ = self.channel_models[0].calculate_pathloss(d, los_condition='LOS')
            pl_nlos[i], _ = self.channel_models[0].calculate_pathloss(d, los_condition='NLOS')
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances, pl_los, 'b-', linewidth=2, label='LOS')
        plt.plot(distances, pl_nlos, 'r--', linewidth=2, label='NLOS')
        plt.xlabel('Distância (m)')
        plt.ylabel('Pathloss (dB)')
        plt.title(f'Modelo de Pathloss 3GPP - Banda n78 ({self.channel_models[0].fc} GHz)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_cell_coverage(self):
        """Visualiza cobertura das células"""
        x = np.linspace(-1000, 1000, 100)
        y = np.linspace(-1000, 1000, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calcula melhor SNR para cada ponto
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                snr_values = []
                for cell_idx in range(self.n_cells):
                    distance = np.sqrt((X[i,j] - self.bs_positions[cell_idx,0])**2 + 
                                     (Y[i,j] - self.bs_positions[cell_idx,1])**2)
                    if distance <= self.cell_radius:
                        info = self.channel_models[cell_idx].calculate_snr(
                            self.tx_power, distance, self.bandwidths[cell_idx])
                        snr_values.append(info['snr_db'])
                if snr_values:
                    Z[i,j] = max(snr_values)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='SNR (dB)')
        plt.scatter(self.bs_positions[:,0], self.bs_positions[:,1], 
                   c='red', s=200, marker='^', label='Estação Base')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Mapa de Cobertura SNR - 5 Células')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# Exemplo de uso
def main():
    # Cria simulador
    simulator = NetworkSimulator(
        n_cells=5,
        bandwidths=[10, 10, 15, 15, 20],  # MHz
        cell_radius=300.0,  # metros
        tx_power=33.0  # dBm
    )
    
    # 1. Plota curvas de pathloss
    print("Gerando curvas de pathloss...")
    simulator.plot_pathloss_curves()
    
    # 2. Plota mapa de cobertura
    print("Gerando mapa de cobertura...")
    simulator.plot_cell_coverage()
    
    # 3. Simula conexões URLLC
    print("\nSimulando conexões URLLC:")
    print("-" * 80)
    
    for cell_id in range(5):
        # Simula UE em posição aleatória dentro da célula
        distance = np.random.uniform(50, simulator.cell_radius)
        
        # Executa simulação
        result = simulator.simulate_urllc_connection(
            cell_id=cell_id,
            ue_distance=distance,
            urllc_bw_requirement=2.0  # Mbps
        )
        
        print(f"Célula {cell_id} ({simulator.bandwidths[cell_id]} MHz):")
        print(f"  Distância UE: {distance:.1f} m")
        print(f"  Condição: {result['los_condition']}")
        print(f"  SNR: {result['snr_db']:.1f} dB")
        print(f"  BER: {result['ber']:.2e}")
        print(f"  Capacidade disponível: {result['available_capacity_mbps']:.1f} Mbps")
        print(f"  Latência total: {result['total_latency_ms']:.2f} ms")
        print(f"  Probabilidade de sucesso: {result['success_probability']:.3f}")
        print()

if __name__ == "__main__":
    main()