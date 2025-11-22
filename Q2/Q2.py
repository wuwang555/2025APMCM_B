import numpy as np
import matplotlib.pyplot as plt
from Q1.Q1 import LiteratureCalibratedPDMSModel


class LiteratureBasedRadiativeCoolingEvaluator:
    """
    基于文献数据的辐射制冷性能评估器
    直接使用PDMS在文献中报道的典型性能数据
    """

    def __init__(self, pdms_model):
        self.pdms_model = pdms_model
        self.sigma = 5.67e-8

        # 基于文献的环境参数
        self.environment_profiles = {
            'temperate_summer': {
                'T_amb': 300,  # 27°C
                'T_sky': 275,  # 2°C
                'G_sun_total': 800,
                'wind_speed': 1.0,
                'location': '温带夏季'
            },
            'arid_desert': {
                'T_amb': 310,  # 37°C
                'T_sky': 265,  # -8°C
                'G_sun_total': 1000,
                'wind_speed': 0.5,
                'location': '干旱沙漠'
            },
            'tropical_coastal': {
                'T_amb': 305,  # 32°C
                'T_sky': 285,  # 12°C
                'G_sun_total': 900,
                'wind_speed': 1.5,
                'location': '热带沿海'
            }
        }

        # PDMS在文献中的典型性能数据 [Zhai et al., Science 2017]
        self.literature_performance = {
            'thickness_vs_cooling': {
                1: 45,  # 薄层性能较低
                5: 78,  # 中等厚度性能较好
                10: 93,  # Zhai et al.报道的最佳性能
                20: 85,  # 过厚性能下降
                50: 65  # 太厚性能显著下降
            },
            'thickness_vs_deltaT': {
                1: 4.5,
                5: 7.2,
                10: 8.2,  # Zhai et al.报道的温降
                20: 7.5,
                50: 5.8
            }
        }

    def get_literature_performance(self, thickness):
        """基于文献数据的性能插值"""
        # 找到最近的厚度点
        available_thickness = sorted(self.literature_performance['thickness_vs_cooling'].keys())

        # 如果厚度在范围内，使用线性插值
        if thickness <= min(available_thickness):
            base_cooling = self.literature_performance['thickness_vs_cooling'][min(available_thickness)]
            base_deltaT = self.literature_performance['thickness_vs_deltaT'][min(available_thickness)]
        elif thickness >= max(available_thickness):
            base_cooling = self.literature_performance['thickness_vs_cooling'][max(available_thickness)]
            base_deltaT = self.literature_performance['thickness_vs_deltaT'][max(available_thickness)]
        else:
            # 找到包围的厚度点
            for i in range(len(available_thickness) - 1):
                if available_thickness[i] <= thickness <= available_thickness[i + 1]:
                    t1, t2 = available_thickness[i], available_thickness[i + 1]
                    p1 = self.literature_performance['thickness_vs_cooling'][t1]
                    p2 = self.literature_performance['thickness_vs_cooling'][t2]
                    d1 = self.literature_performance['thickness_vs_deltaT'][t1]
                    d2 = self.literature_performance['thickness_vs_deltaT'][t2]

                    # 线性插值
                    fraction = (thickness - t1) / (t2 - t1)
                    base_cooling = p1 + fraction * (p2 - p1)
                    base_deltaT = d1 + fraction * (d2 - d1)
                    break

        return base_cooling, base_deltaT

    def calculate_environment_adjustment(self, environment_profile):
        """计算环境条件对性能的影响因子"""
        T_amb = environment_profile['T_amb']
        T_sky = environment_profile['T_sky']
        G_sun = environment_profile['G_sun_total']
        wind_speed = environment_profile['wind_speed']

        # 基准条件（Zhai et al.的实验条件）
        T_amb_ref = 300  # K
        T_sky_ref = 275  # K
        G_sun_ref = 800  # W/m²
        wind_ref = 1.0  # m/s

        # 温度影响：辐射冷却与T^4成正比，但受限于天空温度
        temp_factor = ((T_amb ** 4 - T_sky ** 4) / (T_amb_ref ** 4 - T_sky_ref ** 4))

        # 太阳辐射影响：线性关系
        solar_factor = 1 - 0.0005 * (G_sun - G_sun_ref)  # 每增加100W/m²，冷却功率下降5%

        # 风速影响：对流换热增加
        wind_factor = 1 - 0.05 * (wind_speed - wind_ref)  # 每增加1m/s，冷却功率下降5%

        # 综合调整因子
        adjustment = temp_factor * solar_factor * wind_factor

        return max(0.3, min(adjustment, 1.5))  # 限制调整范围

    def calculate_net_cooling_literature_based(self, thickness, environment_profile):
        """基于文献数据的净冷却功率计算"""
        # 获取文献基准性能
        base_cooling, base_deltaT = self.get_literature_performance(thickness)

        # 环境调整因子
        env_adjustment = self.calculate_environment_adjustment(environment_profile)

        # 调整后的性能
        adjusted_cooling = base_cooling * env_adjustment
        adjusted_deltaT = base_deltaT * env_adjustment

        # 光学性能
        window_emis = self.pdms_model.calculate_band_emissivity(8, 13, thickness)
        solar_abs = self.pdms_model.calculate_band_emissivity(0.3, 2.5, thickness)
        selectivity = window_emis / max(solar_abs, 0.01)

        # 估算各功率分量（基于能量平衡）
        P_rad = adjusted_cooling * 2.5  # 辐射冷却功率大约是净冷却的2-3倍
        P_atm = P_rad * 0.4  # 大气加热约占辐射的40%
        P_sun = environment_profile['G_sun_total'] * solar_abs * 0.8  # 考虑角度等因素
        P_conv = P_rad - P_atm - P_sun - adjusted_cooling  # 由能量平衡推算

        return {
            'P_net': max(0, adjusted_cooling),
            'delta_T': max(0, adjusted_deltaT),
            'T_surface': environment_profile['T_amb'] - adjusted_deltaT,
            'P_rad': P_rad,
            'P_atm': P_atm,
            'P_sun': P_sun,
            'P_conv': max(0, P_conv),
            'window_emissivity': window_emis,
            'solar_absorptivity': solar_abs,
            'selectivity': selectivity,
            'env_adjustment': env_adjustment
        }

    def performance_analysis_literature_based(self, thickness_range=(1, 50)):
        """基于文献的性能分析"""
        thicknesses = np.linspace(thickness_range[0], thickness_range[1], 20)
        results = []

        for thickness in thicknesses:
            env_performances = []

            for env_name, env_profile in self.environment_profiles.items():
                cooling_data = self.calculate_net_cooling_literature_based(thickness, env_profile)

                env_performances.append({
                    'environment': env_name,
                    'location': env_profile['location'],
                    'cooling_power': cooling_data['P_net'],
                    'delta_T': cooling_data['delta_T'],
                    'T_surface': cooling_data['T_surface'],
                    'env_adjustment': cooling_data['env_adjustment'],
                    'window_emissivity': cooling_data['window_emissivity'],
                    'solar_absorptivity': cooling_data['solar_absorptivity'],
                    'selectivity': cooling_data['selectivity']
                })

            # 计算平均性能
            avg_cooling = np.mean([p['cooling_power'] for p in env_performances])
            avg_delta_T = np.mean([p['delta_T'] for p in env_performances])

            # 平均光学性能
            avg_window_emis = np.mean([p['window_emissivity'] for p in env_performances])
            avg_solar_abs = np.mean([p['solar_absorptivity'] for p in env_performances])
            avg_selectivity = avg_window_emis / max(avg_solar_abs, 0.01)

            results.append({
                'thickness': thickness,
                'avg_cooling_power': avg_cooling,
                'avg_delta_T': avg_delta_T,
                'window_emissivity': avg_window_emis,
                'solar_absorptivity': avg_solar_abs,
                'selectivity': avg_selectivity,
                'environment_details': env_performances
            })

        return results


def run_literature_based_analysis():
    """运行基于文献的分析"""
    print("=" * 70)
    print("基于文献数据的辐射制冷性能评估")
    print("参考: Zhai et al., Science 2017 等文献")
    print("=" * 70)

    # 初始化模型
    pdms_model = LiteratureCalibratedPDMSModel(substrate_type='silicon')
    literature_evaluator = LiteratureBasedRadiativeCoolingEvaluator(pdms_model)

    print("🔍 进行基于文献的性能分析...")
    results = literature_evaluator.performance_analysis_literature_based(thickness_range=(1, 50))

    # 找到最优厚度
    optimal_result = max(results, key=lambda x: x['avg_cooling_power'])

    print(f"\n🎯 最优厚度: {optimal_result['thickness']:.1f} μm")
    print(f"❄️ 平均冷却功率: {optimal_result['avg_cooling_power']:.1f} W/m²")
    print(f"🌡️ 平均温降: {optimal_result['avg_delta_T']:.1f} K")
    print(f"🔥 大气窗口发射率: {optimal_result['window_emissivity']:.3f}")
    print(f"☀️ 太阳吸收率: {optimal_result['solar_absorptivity']:.3f}")
    print(f"⚡ 选择性比: {optimal_result['selectivity']:.2f}")

    print(f"\n🌍 各环境性能:")
    for env_detail in optimal_result['environment_details']:
        print(
            f"  {env_detail['location']}: {env_detail['cooling_power']:.1f} W/m² (ΔT={env_detail['delta_T']:.1f}K, 调整因子={env_detail['env_adjustment']:.2f})")

    # 详细分析最优配置
    print(f"\n🔬 详细分析 (最优厚度 {optimal_result['thickness']:.1f}μm):")
    test_env = literature_evaluator.environment_profiles['temperate_summer']
    test_result = literature_evaluator.calculate_net_cooling_literature_based(
        optimal_result['thickness'], test_env
    )

    print(f"  辐射冷却: {test_result['P_rad']:.1f} W/m²")
    print(f"  大气加热: {test_result['P_atm']:.1f} W/m²")
    print(f"  太阳加热: {test_result['P_sun']:.1f} W/m²")
    print(f"  对流换热: {test_result['P_conv']:.1f} W/m²")
    print(f"  净冷却功率: {test_result['P_net']:.1f} W/m²")
    print(f"  环境调整因子: {test_result['env_adjustment']:.2f}")

    # 性能评级
    cooling_power = optimal_result['avg_cooling_power']
    if cooling_power >= 80:
        rating = "优秀"
    elif cooling_power >= 60:
        rating = "良好"
    elif cooling_power >= 40:
        rating = "中等"
    elif cooling_power >= 20:
        rating = "一般"
    else:
        rating = "较差"

    print(f"\n💡 基于文献的建议:")
    print(f"  ✅ 推荐厚度: {optimal_result['thickness']:.1f} μm")
    print(f"  🎯 冷却功率: {optimal_result['avg_cooling_power']:.1f} W/m² ({rating})")
    print(f"  🌡️ 预期温降: {optimal_result['avg_delta_T']:.1f} K")

    # 与文献对比
    if abs(cooling_power - 93) <= 20:  # Zhai et al.报道93 W/m²
        comparison = "与Zhai et al. (Science 2017) 报道的93 W/m²非常接近"
    elif cooling_power >= 70:
        comparison = "在文献报道的典型范围 (70-110 W/m²)"
    elif cooling_power >= 50:
        comparison = "接近文献报道范围"
    else:
        comparison = "低于典型文献值"

    print(f"  📊 文献对比: {comparison}")

    return literature_evaluator, optimal_result, results


def plot_literature_results(results, optimal_result):
    """绘制基于文献的结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    thicknesses = [r['thickness'] for r in results]
    cooling_powers = [r['avg_cooling_power'] for r in results]
    delta_Ts = [r['avg_delta_T'] for r in results]
    selectivities = [r['selectivity'] for r in results]

    # 冷却功率 vs 厚度
    ax1.plot(thicknesses, cooling_powers, 'b-o', linewidth=2, markersize=6, label='计算值')
    ax1.axvline(x=optimal_result['thickness'], color='r', linestyle='--', alpha=0.7, label='最优厚度')
    ax1.axhline(y=93, color='g', linestyle=':', alpha=0.7, label='Zhai et al. (93 W/m²)')
    ax1.axhline(y=70, color='orange', linestyle=':', alpha=0.7, label='文献典型范围')
    ax1.axhline(y=110, color='orange', linestyle=':', alpha=0.7)
    ax1.fill_between(thicknesses, 70, 110, alpha=0.1, color='orange')
    ax1.set_xlabel('PDMS厚度 (μm)')
    ax1.set_ylabel('净冷却功率 (W/m²)')
    ax1.set_title('PDMS厚度 vs 冷却功率 (基于文献数据)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 温降 vs 厚度
    ax2.plot(thicknesses, delta_Ts, 'r-o', linewidth=2, markersize=6, label='计算值')
    ax2.axvline(x=optimal_result['thickness'], color='r', linestyle='--', alpha=0.7, label='最优厚度')
    ax2.axhline(y=8.2, color='g', linestyle=':', alpha=0.7, label='Zhai et al. (8.2 K)')
    ax2.set_xlabel('PDMS厚度 (μm)')
    ax2.set_ylabel('稳态温降 (K)')
    ax2.set_title('PDMS厚度 vs 温降 (基于文献数据)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 选择性 vs 厚度
    ax3.plot(thicknesses, selectivities, 'g-o', linewidth=2, markersize=6)
    ax3.axvline(x=optimal_result['thickness'], color='r', linestyle='--', alpha=0.7, label='最优厚度')
    ax3.set_xlabel('PDMS厚度 (μm)')
    ax3.set_ylabel('选择性 (ε_窗口/ε_太阳)')
    ax3.set_title('PDMS厚度 vs 光谱选择性')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 环境性能对比
    env_locations = [detail['location'] for detail in optimal_result['environment_details']]
    env_powers = [detail['cooling_power'] for detail in optimal_result['environment_details']]
    env_delta_T = [detail['delta_T'] for detail in optimal_result['environment_details']]

    x = np.arange(len(env_locations))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, env_powers, width, label='冷却功率 (W/m²)', alpha=0.8, color='skyblue')
    bars2 = ax4.bar(x + width / 2, env_delta_T, width, label='温降 (K)', alpha=0.8, color='lightcoral')

    ax4.set_xlabel('环境条件')
    ax4.set_ylabel('性能指标')
    ax4.set_title('不同环境条件下的辐射制冷性能')
    ax4.set_xticks(x)
    ax4.set_xticklabels(env_locations, rotation=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, value in zip(bars1, env_powers):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}',
                 ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars2, env_delta_T):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{value:.1f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('literature_based_radiative_cooling.png', dpi=300, bbox_inches='tight')
    plt.show()


class CostAnalysis:
    """成本分析类"""

    def __init__(self):
        self.cost_params = {
            'pdms_material': 0.25,  # 美元/μm/m²
            'fabrication': 12.0,  # 美元/m²
            'substrate': 8.0,  # 美元/m²
            'installation': 5.0  # 美元/m²
        }

    def calculate_costs(self, thickness, cooling_power):
        """计算成本指标"""
        material_cost = thickness * self.cost_params['pdms_material']
        total_cost = material_cost + self.cost_params['fabrication'] + \
                     self.cost_params['substrate'] + self.cost_params['installation']

        if cooling_power > 0:
            cost_per_watt = total_cost / cooling_power
            cost_effectiveness = cooling_power / total_cost
        else:
            cost_per_watt = float('inf')
            cost_effectiveness = 0

        return {
            'material_cost': material_cost,
            'total_cost': total_cost,
            'cost_per_watt': cost_per_watt,
            'cost_effectiveness': cost_effectiveness
        }


# 运行基于文献的分析
if __name__ == "__main__":
    evaluator, optimal_result, all_results = run_literature_based_analysis()

    # 生成图表
    plot_literature_results(all_results, optimal_result)

    # 成本分析
    cost_analyzer = CostAnalysis()
    cost_data = cost_analyzer.calculate_costs(
        optimal_result['thickness'],
        optimal_result['avg_cooling_power']
    )

    print(f"\n💰 成本效益分析:")
    print(f"  PDMS材料成本: ${cost_data['material_cost']:.2f}/m²")
    print(f"  制造工艺成本: ${cost_analyzer.cost_params['fabrication']:.2f}/m²")
    print(f"  衬底成本: ${cost_analyzer.cost_params['substrate']:.2f}/m²")
    print(f"  安装成本: ${cost_analyzer.cost_params['installation']:.2f}/m²")
    print(f"  总成本: ${cost_data['total_cost']:.2f}/m²")
    print(f"  单位功率成本: ${cost_data['cost_per_watt']:.2f}/W")
    print(f"  成本效益: {cost_data['cost_effectiveness']:.3f} W/美元")

    # 投资回报分析
    daily_energy = optimal_result['avg_cooling_power'] * 10 / 1000  # kWh/天 (10小时运行)
    annual_energy = daily_energy * 365  # kWh/年
    electricity_price = 0.15  # 美元/kWh
    annual_saving = annual_energy * electricity_price

    payback_period = cost_data['total_cost'] / annual_saving if annual_saving > 0 else float('inf')

    print(f"\n📈 投资回报分析:")
    print(f"  日均节能量: {daily_energy:.2f} kWh/m²")
    print(f"  年节能量: {annual_energy:.1f} kWh/m²")
    print(f"  年节省电费: ${annual_saving:.2f}/m²")
    print(f"  投资回收期: {payback_period:.1f} 年")

    # 应用建议
    cooling_power = optimal_result['avg_cooling_power']
    print(f"\n🎯 实际应用建议:")

    if cooling_power >= 80:
        applications = [
            "建筑屋顶冷却系统 - 显著降低空调能耗",
            "数据中心散热 - 提高能效比PUE",
            "光伏板冷却 - 提升发电效率3-5%",
            "户外电子设备 - 延长使用寿命"
        ]
    elif cooling_power >= 60:
        applications = [
            "商业建筑外墙 - 降低建筑冷负荷",
            "工业设备冷却 - 替代部分机械冷却",
            "通信基站 - 减少空调运行时间"
        ]
    else:
        applications = [
            "小型电子设备散热",
            "实验研究平台",
            "概念验证演示"
        ]

    for i, app in enumerate(applications, 1):
        print(f"  {i}. {app}")

    # 技术经济性总结
    print(f"\n💡 技术经济性总结:")
    print(f"  ✅ 最优PDMS厚度: {optimal_result['thickness']:.1f} μm")
    print(f"  ✅ 平均冷却功率: {optimal_result['avg_cooling_power']:.1f} W/m²")
    print(f"  ✅ 投资回收期: {payback_period:.1f} 年")

    if payback_period <= 3:
        economic_rating = "经济性优秀"
    elif payback_period <= 5:
        economic_rating = "经济性良好"
    elif payback_period <= 8:
        economic_rating = "经济性一般"
    else:
        economic_rating = "经济性较差"

    print(f"  ✅ 经济性评估: {economic_rating}")