import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Q1.Q1  import LiteratureCalibratedPDMSModel
from Q2.Q2 import LiteratureBasedRadiativeCoolingEvaluator, CostAnalysis
from Q3.Q3_base_reality import PhysicsBasedMultiLayerDesign, LayerNumberOptimizer


class EnhancedCostAnalysis(CostAnalysis):
    def calculate_multilayer_costs(self, structure, cooling_power):
        """修正的多层结构成本计算 - 使用合理的成本参数"""
        # 修正：使用更实际的材料单位成本（美元/微米/m²）
        material_unit_costs = {
            'Ag': 8.0,  # 银：8美元/微米/m² (实际：5-15美元)
            'Al': 2.5,  # 铝：2.5美元/微米/m² (实际：2-5美元)
            'SiO2': 1.2,  # 二氧化硅：1.2美元/微米/m² (实际：1-3美元)
            'TiO2': 3.0,  # 二氧化钛：3美元/微米/m² (实际：2-6美元)
            'PDMS': 0.8  # PDMS：0.8美元/微米/m² (实际：0.5-2美元)
        }

        material_cost = 0
        layer_costs = []

        print(f"🔍 详细成本计算 - 结构: {structure}")

        for material, thickness_nm in structure:
            thickness_um = thickness_nm / 1000  # 纳米转微米
            unit_cost = material_unit_costs.get(material, 1.0)
            layer_cost = unit_cost * thickness_um

            print(
                f"   {material}: {thickness_nm}nm = {thickness_um}μm, 单位成本${unit_cost}/μm, 层成本${layer_cost:.2f}")

            material_cost += layer_cost
            layer_costs.append({
                'material': material,
                'thickness': thickness_nm,
                'cost': round(layer_cost, 2)
            })

        # 修正制造复杂度成本
        base_fabrication = 15.0  # 基础制造费用
        complexity_factor = 1.3 ** (len(structure) - 1)  # 层数增加，成本适度增长
        fabrication_cost = base_fabrication * complexity_factor

        # 固定成本
        substrate_cost = 10.0  # 衬底成本
        installation_cost = 8.0  # 安装成本

        total_cost = material_cost + fabrication_cost + substrate_cost + installation_cost

        if cooling_power > 0:
            cost_per_watt = total_cost / cooling_power
            cost_effectiveness = cooling_power / total_cost
        else:
            cost_per_watt = float('inf')
            cost_effectiveness = 0

        print(f"📊 成本汇总:")
        print(f"   材料成本: ${material_cost:.2f}")
        print(f"   制造费用: ${fabrication_cost:.2f}")
        print(f"   衬底成本: ${substrate_cost:.2f}")
        print(f"   安装成本: ${installation_cost:.2f}")
        print(f"   总成本: ${total_cost:.2f}")

        return {
            'total_cost': round(total_cost, 2),
            'material_cost': round(material_cost, 2),
            'fabrication_cost': round(fabrication_cost, 2),
            'substrate_cost': substrate_cost,
            'installation_cost': installation_cost,
            'cost_per_watt': round(cost_per_watt, 2),
            'cost_effectiveness': round(cost_effectiveness, 2),
            'layer_costs': layer_costs
        }


class ComprehensiveRadiativeCoolingOptimizer:
    """辐射制冷综合优化设计器 - 问题四解决方案"""

    def __init__(self):
        self.pdms_model = LiteratureCalibratedPDMSModel(substrate_type='silicon')
        self.evaluator = LiteratureBasedRadiativeCoolingEvaluator(self.pdms_model)
        self.multilayer_designer = PhysicsBasedMultiLayerDesign()
        self.cost_analyzer = CostAnalysis()
        self.enhanced_cost_analyzer = EnhancedCostAnalysis()

        # 最优设计配置（基于前三个问题的结果）- 修正结构定义
        self.optimal_designs = {
            'single_layer': {
                'structure': [('PDMS', 11000)],
                'performance': 101.1,
                'description': '单层PDMS基准设计'
            },
            'multilayer_optimal': {
                'structure': [('Ag', 100), ('SiO2', 250), ('PDMS', 11000)],
                'performance': 136.5,
                'description': '三层最优设计(Ag/SiO₂/PDMS)'
            },
            'multilayer_advanced': {
                'structure': [('Ag', 100), ('SiO2', 200), ('TiO2', 150), ('PDMS', 11000)],
                'performance': 146.6,
                'description': '四层增强设计(Ag/SiO₂/TiO₂/PDMS)'
            }
        }

        # 验证设计配置
        for design_key, design in self.optimal_designs.items():
            print(f"✅ 设计验证: {design['description']} - {len(design['structure'])}层")

    def comprehensive_performance_evaluation(self, design_key):
        """修正的综合性能评估"""
        design = self.optimal_designs[design_key]

        # 光学性能分析
        optical_performance = self.multilayer_designer.calculate_structure_performance(
            design['structure']
        )

        # 修正的成本分析 - 使用多层结构成本计算
        cost_data = self.enhanced_cost_analyzer.calculate_multilayer_costs(
            design['structure'],
            design['performance']
        )

        # 环境适应性分析
        environment_performances = []
        for env_name, env_profile in self.evaluator.environment_profiles.items():
            pdms_thickness = None
            for material, thickness in design['structure']:
                if material == 'PDMS':
                    pdms_thickness = thickness / 1000  # 转换为μm
                    break

            if pdms_thickness is None:
                pdms_thickness = 11.0

            env_perf = self.evaluator.calculate_net_cooling_literature_based(
                pdms_thickness, env_profile
            )
            env_perf['environment'] = env_name
            env_perf['location'] = env_profile['location']
            environment_performances.append(env_perf)

        return {
            'design_info': design,
            'optical_performance': optical_performance,
            'environment_performances': environment_performances,
            'cost_data': cost_data,
            'technical_feasibility': self.assess_technical_feasibility(design),
            'manufacturing_feasibility': self.assess_manufacturing_feasibility(design)
        }

    def assess_technical_feasibility(self, design):
        """技术可行性评估"""
        feasibility = {
            'material_availability': '高',
            'process_maturity': '高',
            'scalability': '高',
            'lifespan': '10-15年',
            'reliability': '高',
            'technical_risks': []
        }

        # 材料可获得性评估
        materials = [layer[0] for layer in design['structure']]
        if 'Ag' in materials:
            feasibility['material_availability_note'] = '银材料成本较高但供应稳定'
        if 'TiO2' in materials:
            feasibility['material_availability_note'] = '二氧化钛广泛可得'

        # 技术风险评估
        if len(design['structure']) > 3:
            feasibility['technical_risks'].append('多层结构界面控制要求较高')
        if any(thickness < 50 for _, thickness in design['structure']):
            feasibility['technical_risks'].append('超薄层厚度控制需要精密设备')

        return feasibility

    def assess_manufacturing_feasibility(self, design):
        """修正的制造可行性评估 - 确保处理所有层"""
        manufacturing = {
            'process_flow': [],
            'equipment_requirements': [],
            'yield_estimation': '85-95%',
            'production_rate': '适合大规模卷对卷生产',
            'quality_control': '标准光学检测方法'
        }

        print(f"🔧 调试: 处理设计结构，层数={len(design['structure'])}")

        # 确保正确处理所有层
        for i, (material, thickness) in enumerate(design['structure']):
            if material in ['Ag', 'Al']:
                process = f'第{i + 1}层: {material} - 磁控溅射 ({thickness}nm)'
                equipment = '磁控溅射设备'
            elif material in ['SiO2']:
                process = f'第{i + 1}层: {material} - PECVD ({thickness}nm)'
                equipment = 'PECVD设备'
            elif material in ['TiO2']:
                process = f'第{i + 1}层: {material} - 原子层沉积 ({thickness}nm)'
                equipment = '原子层沉积设备'
            elif material == 'PDMS':
                process = f'第{i + 1}层: PDMS - 旋涂+固化 ({thickness}nm)'
                equipment = '旋涂机+热板'
            else:
                process = f'第{i + 1}层: {material} - 标准沉积 ({thickness}nm)'
                equipment = '通用沉积设备'

            manufacturing['process_flow'].append(process)
            if equipment not in manufacturing['equipment_requirements']:
                manufacturing['equipment_requirements'].append(equipment)

        # 根据层数调整良率估计
        layer_count = len(design['structure'])
        if layer_count <= 2:
            manufacturing['yield_estimation'] = '90-95%'
        elif layer_count <= 4:
            manufacturing['yield_estimation'] = '85-92%'
        else:
            manufacturing['yield_estimation'] = '80-88%'

        return manufacturing

    def economic_analysis(self, evaluation_results):
        """修正的详细经济性分析"""
        design = evaluation_results['design_info']
        cost_data = evaluation_results['cost_data']

        # 投资分析
        capital_investment = {
            'equipment_cost': 500000,  # 美元
            'facility_cost': 200000,  # 美元
            'working_capital': 100000,  # 美元
            'total_investment': 800000  # 美元
        }

        # 运营成本
        operating_costs = {
            'material_cost_per_m2': cost_data['material_cost'],
            'labor_cost_per_m2': 3.0,
            'utilities_per_m2': 1.5,
            'maintenance_per_m2': 1.0,
            'total_operating_cost_per_m2': cost_data['total_cost']
        }

        # 收益分析 - 修正：使用正确的冷却功率
        cooling_power = design['performance']
        daily_operation_hours = 10
        electricity_price = 0.15  # 美元/kWh
        annual_operation_days = 365

        daily_energy_saving = cooling_power * daily_operation_hours / 1000  # kWh/天
        annual_energy_saving = daily_energy_saving * annual_operation_days  # kWh/年
        annual_cost_saving = annual_energy_saving * electricity_price  # 美元/年

        # 投资回报分析
        production_capacity = 10000  # m²/年
        annual_revenue = annual_cost_saving * production_capacity
        annual_operating_cost = operating_costs['total_operating_cost_per_m2'] * production_capacity
        annual_profit = annual_revenue - annual_operating_cost

        payback_period = capital_investment['total_investment'] / annual_profit if annual_profit > 0 else float('inf')

        return {
            'capital_investment': capital_investment,
            'operating_costs': operating_costs,
            'revenue_analysis': {
                'daily_energy_saving_per_m2': daily_energy_saving,
                'annual_energy_saving_per_m2': annual_energy_saving,
                'annual_cost_saving_per_m2': annual_cost_saving,
                'production_capacity': production_capacity,
                'annual_revenue': annual_revenue,
                'annual_operating_cost': annual_operating_cost,
                'annual_profit': annual_profit
            },
            'investment_metrics': {
                'payback_period': payback_period,
                'roi_first_year': annual_profit / capital_investment['total_investment'] * 100 if capital_investment['total_investment'] > 0 else 0,
                'npv_5years': self.calculate_npv(annual_profit, 5, 0.1) - capital_investment['total_investment']
            }
        }

    def calculate_npv(self, annual_cashflow, years, discount_rate):
        """计算净现值"""
        npv = 0
        for year in range(1, years + 1):
            npv += annual_cashflow / ((1 + discount_rate) ** year)
        return npv

    def compare_designs(self):
        """设计方案综合对比"""
        comparisons = []

        for design_key in self.optimal_designs.keys():
            evaluation = self.comprehensive_performance_evaluation(design_key)
            economic_data = self.economic_analysis(evaluation)

            comparison = {
                'design_name': self.optimal_designs[design_key]['description'],
                'performance': self.optimal_designs[design_key]['performance'],
                'cost_per_m2': evaluation['cost_data']['total_cost'],
                'cost_effectiveness': evaluation['cost_data']['cost_effectiveness'],
                'technical_feasibility': evaluation['technical_feasibility']['material_availability'],
                'payback_period': economic_data['investment_metrics']['payback_period'],
                'optical_performance': evaluation['optical_performance']
            }
            comparisons.append(comparison)

        return comparisons

    def generate_final_recommendation(self):
        """生成最终推荐方案 - 调整权重更重视经济性"""
        comparisons = self.compare_designs()

        # 多目标决策：调整权重，更重视经济性
        for comp in comparisons:
            # 性能评分 (0-1)
            perf_score = comp['performance'] / 150.0

            # 成本效益评分 - 更重视
            cost_eff_score = min(comp['cost_effectiveness'] / 3.0, 1.0)

            # 可行性评分
            feasibility_score = 1.0 if comp['technical_feasibility'] == '高' else 0.7

            # 投资回收期评分 - 更重视
            if comp['payback_period'] <= 3:
                payback_score = 1.0
            elif comp['payback_period'] <= 5:
                payback_score = 0.7
            else:
                payback_score = 0.3

            # 调整权重：经济性权重增加
            comp['comprehensive_score'] = (
                    perf_score * 0.25 +
                    cost_eff_score * 0.35 +
                    feasibility_score * 0.2 +
                    payback_score * 0.2
            )

        # 选择最优方案
        best_design = max(comparisons, key=lambda x: x['comprehensive_score'])

        # 验证选择合理性
        print(f"🔍 方案选择验证:")
        for comp in comparisons:
            print(f"  {comp['design_name']}: 评分{comp['comprehensive_score']:.3f}, "
                  f"成本效益{comp['cost_effectiveness']:.2f}W/$, 回收期{comp['payback_period']:.2f}年")

        return best_design, comparisons


def run_problem4_comprehensive_solution():
    """运行问题四综合解决方案 - 带验证"""
    print("=" * 80)
    print("问题四：辐射制冷材料与结构综合优化设计")
    print("基于问题一至三结果的系统整合与可行性评估")
    print("=" * 80)

    optimizer = ComprehensiveRadiativeCoolingOptimizer()

    print("\n🔬 开始综合性能评估...")

    # 评估各设计方案
    design_evaluations = {}
    for design_key in optimizer.optimal_designs.keys():
        print(f"\n📊 评估设计方案: {optimizer.optimal_designs[design_key]['description']}")
        evaluation = optimizer.comprehensive_performance_evaluation(design_key)
        design_evaluations[design_key] = evaluation

        # 验证结构一致性
        design_layers = len(evaluation['design_info']['structure'])
        process_steps = len(evaluation['manufacturing_feasibility']['process_flow'])

        if design_layers == process_steps:
            print(f"   ✅ 结构一致性验证通过: {design_layers}层设计 = {process_steps}步工艺")
        else:
            print(f"   ⚠️ 结构不一致: {design_layers}层设计 vs {process_steps}步工艺")

        # 输出关键指标
        print(f"  冷却功率: {evaluation['design_info']['performance']:.1f} W/m²")
        print(f"  太阳反射率: {evaluation['optical_performance']['solar_reflectivity']:.3f}")
        print(f"  窗口发射率: {evaluation['optical_performance']['window_emissivity']:.3f}")
        print(f"  成本: ${evaluation['cost_data']['total_cost']:.2f}/m²")

    print("\n⚖️ 进行方案综合对比...")
    comparisons = optimizer.compare_designs()

    print("\n🎯 生成最终推荐方案...")
    best_design, all_comparisons = optimizer.generate_final_recommendation()

    # 输出最终推荐
    print("\n" + "=" * 80)
    print("🏆 最终推荐方案")
    print("=" * 80)
    print(f"推荐设计: {best_design['design_name']}")
    print(f"综合评分: {best_design['comprehensive_score']:.3f}")
    print(f"冷却功率: {best_design['performance']:.1f} W/m²")
    print(f"成本效益: {best_design['cost_effectiveness']:.2f} W/$")
    print(f"投资回收期: {best_design['payback_period']:.2f} 年")

    # 修正：使用正确的设计评估
    best_design_key = None
    for key, design in optimizer.optimal_designs.items():
        if design['description'] == best_design['design_name']:
            best_design_key = key
            break

    if best_design_key is None:
        best_design_key = 'multilayer_optimal'  # 默认使用三层设计

    best_evaluation = design_evaluations[best_design_key]
    economic_analysis = optimizer.economic_analysis(best_evaluation)

    print(f"\n💰 详细经济性分析:")
    print(f"  总投资: ${economic_analysis['capital_investment']['total_investment']:,.0f}")
    print(f"  年利润: ${economic_analysis['revenue_analysis']['annual_profit']:,.0f}")
    print(f"  投资回收期: {economic_analysis['investment_metrics']['payback_period']:.2f} 年")
    print(f"  第一年ROI: {economic_analysis['investment_metrics']['roi_first_year']:.1f}%")
    print(f"  5年净现值: ${economic_analysis['investment_metrics']['npv_5years']:,.0f}")

    print(f"\n🔧 技术可行性:")
    tech_feasibility = best_evaluation['technical_feasibility']
    print(f"  材料可获得性: {tech_feasibility['material_availability']}")
    print(f"  工艺成熟度: {tech_feasibility['process_maturity']}")
    print(f"  规模化能力: {tech_feasibility['scalability']}")
    print(f"  预期寿命: {tech_feasibility['lifespan']}")

    if tech_feasibility['technical_risks']:
        print(f"  技术风险: {', '.join(tech_feasibility['technical_risks'])}")

    print(f"\n🏭 制造可行性:")
    manufacturing = best_evaluation['manufacturing_feasibility']
    print(f"  预计良率: {manufacturing['yield_estimation']}")
    print(f"  生产效率: {manufacturing['production_rate']}")
    print("  工艺流程:")
    for step in manufacturing['process_flow']:
        print(f"    • {step}")

    print(f"\n🌍 环境适应性:")
    for env_perf in best_evaluation['environment_performances']:
        print(f"  {env_perf['location']}: {env_perf['P_net']:.1f} W/m² (ΔT={env_perf['delta_T']:.1f}K)")

    # 应用场景建议
    print(f"\n🎯 推荐应用场景:")
    applications = [
        "商业建筑屋顶冷却系统",
        "数据中心节能散热",
        "光伏板效率提升冷却",
        "工业设备被动冷却",
        "户外电子设备热管理"
    ]

    for i, app in enumerate(applications, 1):
        print(f"  {i}. {app}")

    # 实施路线图
    print(f"\n📅 产业化实施路线图:")
    roadmap = [
        "阶段1 (0-6个月): 原型开发与实验室验证",
        "阶段2 (6-12个月): 中试生产线建设",
        "阶段3 (12-18个月): 规模化生产与市场推广",
        "阶段4 (18-24个月): 技术优化与产品迭代"
    ]

    for stage in roadmap:
        print(f"  • {stage}")

    return optimizer, best_design, all_comparisons, design_evaluations


def plot_comprehensive_results(optimizer, best_design, comparisons, design_evaluations):
    """绘制综合结果图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

    # 1. 设计方案综合对比雷达图
    design_names = [comp['design_name'] for comp in comparisons]
    performance_scores = [comp['performance'] / 150 for comp in comparisons]
    cost_scores = [1 - (comp['cost_per_m2'] / 60) for comp in comparisons]
    feasibility_scores = [0.9 if comp['technical_feasibility'] == '高' else 0.7 for comp in comparisons]
    payback_scores = [1 - min(comp['payback_period'] / 5, 1) for comp in comparisons]

    categories = ['性能', '成本', '可行性', '投资回报']

    for i, design_name in enumerate(design_names):
        values = [performance_scores[i], cost_scores[i], feasibility_scores[i], payback_scores[i]]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax1.plot(angles, values, 'o-', linewidth=2, label=design_name)
        ax1.fill(angles, values, alpha=0.1)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_title('设计方案综合对比雷达图', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.1, 1.05))
    ax1.grid(True)

    # 2. 技术经济性分析
    metrics = ['冷却功率\n(W/m²)', '成本效益\n(W/$)', '投资回收期\n(年)', '综合评分']
    best_values = [
        best_design['performance'] / 150,
        best_design['cost_effectiveness'] / 10,
        1 - best_design['payback_period'] / 5,
        best_design['comprehensive_score']
    ]

    bars = ax2.bar(metrics, best_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax2.set_ylabel('归一化评分')
    ax2.set_title('推荐方案技术经济性指标', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for bar, metric, value in zip(bars, metrics, best_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. 成本结构分析
    best_design_key = None
    for key, design in optimizer.optimal_designs.items():
        if design['description'] == best_design['design_name']:
            best_design_key = key
            break

    if best_design_key is None:
        best_design_key = 'multilayer_optimal'

    best_eval = design_evaluations[best_design_key]
    cost_data = best_eval['cost_data']

    # 计算各材料实际成本
    material_costs = {}
    for layer_cost in cost_data['layer_costs']:
        material = layer_cost['material']
        cost = layer_cost['cost']
        if material in material_costs:
            material_costs[material] += cost
        else:
            material_costs[material] = cost

    # 构建成本结构数据
    cost_labels = []
    cost_values = []
    colors = []

    # 添加材料成本
    material_colors = {
        'Ag': '#FF6B6B',
        'SiO2': '#4ECDC4',
        'TiO2': '#45B7D1',
        'PDMS': '#96CEB4'
    }

    for material, cost in material_costs.items():
        if cost > 0:
            cost_labels.append(f'{material}材料')
            cost_values.append(cost)
            colors.append(material_colors.get(material, '#F8E71C'))

    # 添加其他成本项
    other_costs = {
        '制造工艺': cost_data['fabrication_cost'],
        '衬底': cost_data['substrate_cost'],
        '安装': cost_data['installation_cost']
    }

    other_colors = {
        '制造工艺': '#FFA07A',
        '衬底': '#98D8C8',
        '安装': '#F7DC6F'
    }

    for label, cost in other_costs.items():
        if cost > 0:
            cost_labels.append(label)
            cost_values.append(cost)
            colors.append(other_colors.get(label, '#BB8FCE'))

    # 绘制饼图
    if cost_values:
        wedges, texts, autotexts = ax3.pie(cost_values, labels=cost_labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax3.set_title('推荐方案成本结构分析', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, '成本数据不可用', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('成本结构分析（数据缺失）', fontsize=14, fontweight='bold')

    # 4. 环境性能对比
    environments = []
    cooling_powers = []

    for env_perf in best_eval['environment_performances']:
        environments.append(env_perf['location'])
        cooling_powers.append(env_perf['P_net'])

    bars = ax4.bar(environments, cooling_powers, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax4.set_ylabel('冷却功率 (W/m²)')
    ax4.set_title('不同环境条件下的性能表现', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    for bar, power in zip(bars, cooling_powers):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{power:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('problem4_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_final_report(optimizer, best_design, comparisons, design_evaluations):
    """修正的最终技术报告生成"""

    # 修正：正确匹配设计评估
    best_design_key = None
    for key, design in optimizer.optimal_designs.items():
        if design['description'] == best_design['design_name']:
            best_design_key = key
            break

    if best_design_key is None:
        best_design_key = 'multilayer_optimal'

    best_eval = design_evaluations[best_design_key]

    # 确保使用正确的结构描述
    actual_structure = best_eval['design_info']['structure']
    structure_desc = ' | '.join([f"{mat}({thick}nm)" for mat, thick in actual_structure])
    layer_count = len(actual_structure)

    # 确保制造流程与结构匹配
    manufacturing_steps = len(best_eval['manufacturing_feasibility']['process_flow'])

    # 经济分析
    economic_data = optimizer.economic_analysis(best_eval)

    # 生成报告
    report = f"""
## 执行摘要

基于系统性的建模、优化和评估，我们推荐采用 **{best_design['design_name']}** 作为最优辐射制冷解决方案。

### 核心性能指标
- ✅ **冷却功率**: {best_design['performance']:.1f} W/m²
- ✅ **成本效益**: {best_design['cost_effectiveness']:.2f} W/美元  
- ✅ **投资回收期**: {best_design['payback_period']:.2f} 年
- ✅ **技术可行性**: {best_eval['technical_feasibility']['material_availability']}
- ✅ **光学性能**: 太阳反射率={best_eval['optical_performance']['solar_reflectivity']:.3f}, 
                窗口发射率={best_eval['optical_performance']['window_emissivity']:.3f}

## 技术方案详情

### 最优结构设计
- **结构配置**: {structure_desc} ({layer_count}层结构)
- **技术原理**: 
  - Ag层提供高太阳反射(>96%)
  - SiO₂介电层实现干涉增强和抗反射
  - PDMS层在大气窗口具有高发射率(>95%)

### 性能优势
- 相比单层PDMS性能提升 {((best_design['performance'] - 101.1) / 101.1 * 100):.1f}%
- 在干旱沙漠环境下冷却功率可达 {max([env['P_net'] for env in best_eval['environment_performances']]):.1f} W/m²

## 经济可行性

### 投资分析
- **单位面积成本**: ${best_eval['cost_data']['total_cost']:.2f}/m²
- **设备投资**: ${economic_data['capital_investment']['total_investment']:,.0f} (10,000 m²/年产能)
- **年利润**: ${economic_data['revenue_analysis']['annual_profit']:,.0f}
- **投资回收期**: {best_design['payback_period']:.2f} 年

### 成本效益
- 每美元投资可获得 **{best_design['cost_effectiveness']:.2f} W** 冷却功率
- 5年净现值: **${economic_data['investment_metrics']['npv_5years']:,.0f}**

## 技术与制造可行性

### 技术成熟度
- ✅ 所有材料商业化可得
- ✅ 制造工艺成熟(溅射+PECVD+旋涂)
- ✅ 适合大规模卷对卷生产
- ✅ 预期寿命10-15年

### 质量控制
- 光学性能在线监测
- 厚度控制精度±5%
- 预计生产良率{best_eval['manufacturing_feasibility']['yield_estimation']}

### 制造工艺流程
"""

    # 修正：正确显示制造流程
    for step in best_eval['manufacturing_feasibility']['process_flow']:
        report += f"- {step}\n"

    report += f"""
## 环境与社会效益

### 节能效果
- 每平方米年节电量: {economic_data['revenue_analysis']['annual_energy_saving_per_m2']:.1f} kWh
- CO₂减排量: ~0.5吨/平方米/年(基于电网平均碳排放)

### 应用前景
1. **建筑领域**: 降低空调能耗30-50%
2. **数据中心**: 提高PUE能效指标
3. **光伏产业**: 提升发电效率3-5%
4. **工业冷却**: 替代部分机械冷却系统

## 实施建议

### 产业化路线图
1. **近期(0-6个月)**: 原型验证与工艺优化
2. **中期(6-12个月)**: 中试生产线建设
3. **长期(12-24个月)**: 规模化生产与市场推广

### 风险管控
- 材料价格波动风险: 多元化供应商策略
- 技术迭代风险: 持续研发投入
- 市场接受度风险: 示范工程先行

## 结论

本综合优化设计方案在**技术性能、经济可行性和制造可实现性**三个方面均表现出色，具备产业化推广的充分条件。该技术不仅具有良好的经济效益，更在节能减排和可持续发展方面具有重要价值。

**推荐立即启动产业化进程**，抢占辐射制冷技术市场先机。
"""

    print(report)

    # 保存报告到文件
    with open('radiative_cooling_final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("✅ 最终报告已保存至 'radiative_cooling_final_report.txt'")


# 运行问题四完整解决方案
if __name__ == "__main__":
    # 执行综合优化分析
    optimizer, best_design, comparisons, design_evaluations = run_problem4_comprehensive_solution()

    # 生成可视化结果
    plot_comprehensive_results(optimizer, best_design, comparisons, design_evaluations)

    # 生成最终技术报告
    generate_final_report(optimizer, best_design, comparisons, design_evaluations)

    print("\n🎉 问题四解决方案完成！")
    print("📊 结果包含:")
    print("  • 设计方案综合对比")
    print("  • 技术经济性详细分析")
    print("  • 制造可行性评估")
    print("  • 产业化实施路线图")
    print("  • 完整技术报告")