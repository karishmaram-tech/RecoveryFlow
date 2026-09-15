"""
RecoveryFlow Synthetic Data Generation

Generate realistic payment recovery dataset with documented assumptions.
All assumptions based on published payment industry data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import json
from datetime import datetime, timedelta


# ============================================================================
# ASSUMPTIONS DOCUMENTATION
# ============================================================================

ASSUMPTIONS = {
    "data_source": "Synthetic, based on industry statistics",
    "payment_industry_sources": [
        "Stripe payment intelligence reports",
        "Razorpay merchant analytics",
        "Justt payment recovery benchmarks",
        "Chargeback.com industry statistics"
    ],
    
    "failure_rate": {
        "value": 0.03,
        "range": "2-5% (industry average)",
        "source": "Published payment processor reports",
    },
    
    "recovery_rates_by_strategy": {
        "sms": {
            "rate": 0.82,
            "range": "70-90%",
            "source": "Silverflow, Justt benchmarks",
        },
        "email": {
            "rate": 0.55,
            "range": "40-65%",
            "source": "Payment processor retry reports",
        },
        "delayed_retry": {
            "rate": 0.72,
            "range": "60-80%",
            "source": "Razorpay retry mechanics research",
        },
        "support": {
            "rate": 0.91,
            "range": "85-95%",
            "source": "Direct support team statistics",
        }
    },
    
    "chargeback_rate": {
        "baseline": 0.015,
        "range": "1-2% (industry average)",
        "source": "Visa, Mastercard chargeback statistics",
    },
    
    "customer_segments": {
        "high_ltv_stable": {
            "proportion": 0.20,
            "recovery_baseline": 0.75,
            "chargeback_rate": 0.005,
        },
        "mid_ltv_transient": {
            "proportion": 0.50,
            "recovery_baseline": 0.55,
            "chargeback_rate": 0.02,
        },
        "low_ltv_atrisk": {
            "proportion": 0.30,
            "recovery_baseline": 0.35,
            "chargeback_rate": 0.05,
        }
    }
}


@dataclass
class SyntheticPaymentData:
    """Complete synthetic payment recovery dataset"""
    failures: List[Dict]
    metadata: Dict
    statistics: Dict


class RealisticDataGenerator:
    """Generate synthetic payment data matching real-world patterns"""
    
    def __init__(self, seed: int = 42, n_merchants: int = 50, n_customers_per_merchant: int = 200):
        np.random.seed(seed)
        self.seed = seed
        self.n_merchants = n_merchants
        self.n_customers_per_merchant = n_customers_per_merchant
        self.n_customers = n_merchants * n_customers_per_merchant
    
    def generate(self) -> SyntheticPaymentData:
        """Generate complete realistic dataset"""
        failures = []
        customer_data = self._generate_customers()
        
        base_date = datetime(2024, 1, 1)
        
        for month in range(12):
            current_date = base_date + timedelta(days=30 * month)
            
            for customer_id, customer in customer_data.items():
                if np.random.random() < ASSUMPTIONS["failure_rate"]["value"]:
                    failure = self._generate_failure(customer_id, customer, current_date, month)
                    failures.append(failure)
        
        stats = self._calculate_statistics(failures)
        
        return SyntheticPaymentData(
            failures=failures,
            metadata={
                "seed": self.seed,
                "n_merchants": self.n_merchants,
                "n_customers": self.n_customers,
                "months_simulated": 12,
                "total_failures": len(failures),
                "generated_at": datetime.now().isoformat(),
            },
            statistics=stats,
        )
    
    def _generate_customers(self) -> Dict:
        """Generate customer profiles"""
        customers = {}
        
        for i in range(self.n_customers):
            segment_rand = np.random.random()
            if segment_rand < 0.20:
                segment = "high_ltv_stable"
            elif segment_rand < 0.70:
                segment = "mid_ltv_transient"
            else:
                segment = "low_ltv_atrisk"
            
            if segment == "high_ltv_stable":
                tenure = max(1, int(np.random.normal(36, 12)))
                value = max(500, np.random.lognormal(6.5, 0.8))
            elif segment == "mid_ltv_transient":
                tenure = max(1, int(np.random.normal(12, 8)))
                value = max(100, np.random.lognormal(5.8, 0.7))
            else:
                tenure = max(1, int(np.random.normal(3, 2)))
                value = max(50, np.random.lognormal(4.8, 0.6))
            
            customers[f"cust_{i}"] = {
                "segment": segment,
                "tenure_months": tenure,
                "monthly_value": float(value),
                "backup_methods": np.random.randint(1, 4),
                "chargeback_count": int(np.random.poisson(tenure * 0.01)),
                "subscriptions": [f"sub_{i}_{s}" for s in range(np.random.randint(1, 3))],
            }
        
        return customers
    
    def _generate_failure(self, customer_id: str, customer: Dict, date: datetime, month: int) -> Dict:
        """Generate single payment failure"""
        segment = customer["segment"]
        value = customer["monthly_value"]
        tenure = customer["tenure_months"]
        
        failure_reason = np.random.choice(
            ["card_expired", "insufficient_funds", "timeout", "fraud_blocked", "invalid_account"],
            p=[0.35, 0.30, 0.20, 0.10, 0.05]
        )
        
        is_temporary = failure_reason in ["card_expired", "insufficient_funds", "timeout"]
        
        if segment == "high_ltv_stable":
            strategy = np.random.choice(["sms", "email", "delayed_retry", "support"], p=[0.4, 0.2, 0.2, 0.2])
        elif segment == "mid_ltv_transient":
            strategy = np.random.choice(["sms", "email", "delayed_retry", "support"], p=[0.5, 0.3, 0.15, 0.05])
        else:
            strategy = np.random.choice(["sms", "email", "delayed_retry", "support"], p=[0.4, 0.5, 0.08, 0.02])
        
        strategy_base_rates = {"sms": 0.82, "email": 0.55, "delayed_retry": 0.72, "support": 0.91}
        segment_adjustment = {"high_ltv_stable": 1.1, "mid_ltv_transient": 1.0, "low_ltv_atrisk": 0.7}
        
        base_rate = strategy_base_rates[strategy]
        adjusted_rate = np.clip(base_rate * segment_adjustment[segment], 0.1, 0.95)
        was_recovered = np.random.random() < adjusted_rate
        
        if was_recovered:
            if strategy == "sms":
                time_to_recovery = max(10, int(np.random.normal(45, 15)))
            elif strategy == "email":
                time_to_recovery = max(60, int(np.random.normal(3600, 1800)))
            else:
                time_to_recovery = max(60, int(np.random.normal(7200, 3600)))
        else:
            time_to_recovery = None
        
        churned = False
        if not was_recovered:
            churned = np.random.random() < 0.95
        
        return {
            "id": f"fail_{customer_id}_{month}",
            "customer_id": customer_id,
            "date": date.isoformat(),
            "month": month,
            "amount": value,
            "failure_reason": failure_reason,
            "is_temporary": is_temporary,
            "recovery_strategy": strategy,
            "was_recovered": was_recovered,
            "time_to_recovery_seconds": time_to_recovery,
            "customer_churned_30d": churned,
            "customer_segment": segment,
            "customer_tenure": tenure,
            "backup_methods": customer["backup_methods"],
            "chargeback_history": customer["chargeback_count"],
        }
    
    def _calculate_statistics(self, failures: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        if not failures:
            return {}
        
        recovered = [f for f in failures if f["was_recovered"]]
        
        stats = {
            "total_failures": len(failures),
            "recovered": len(recovered),
            "overall_recovery_rate": len(recovered) / len(failures),
            "total_revenue_at_risk": sum(f["amount"] for f in failures),
            "total_revenue_recovered": sum(f["amount"] for f in recovered),
            "by_strategy": {},
            "by_segment": {},
        }
        
        for strategy in ["sms", "email", "delayed_retry", "support"]:
            s_cases = [f for f in failures if f["recovery_strategy"] == strategy]
            if s_cases:
                s_recovered = len([f for f in s_cases if f["was_recovered"]])
                stats["by_strategy"][strategy] = {
                    "attempts": len(s_cases),
                    "recovered": s_recovered,
                    "recovery_rate": s_recovered / len(s_cases),
                }
        
        for segment in ["high_ltv_stable", "mid_ltv_transient", "low_ltv_atrisk"]:
            seg_cases = [f for f in failures if f["customer_segment"] == segment]
            if seg_cases:
                seg_recovered = len([f for f in seg_cases if f["was_recovered"]])
                stats["by_segment"][segment] = {
                    "failures": len(seg_cases),
                    "recovered": seg_recovered,
                    "recovery_rate": seg_recovered / len(seg_cases),
                    "total_at_risk": sum(f["amount"] for f in seg_cases),
                    "total_recovered": sum(f["amount"] for f in seg_cases if f["was_recovered"]),
                }
        
        return stats
    
    def export_to_json(self, filename: str) -> None:
        """Export dataset to JSON"""
        data = self.generate()
        with open(filename, 'w') as f:
            json.dump({
                'failures': data.failures,
                'metadata': data.metadata,
                'statistics': data.statistics,
                'assumptions': ASSUMPTIONS,
            }, f, indent=2)
    
    def validate_assumptions(self, data: SyntheticPaymentData) -> Dict[str, bool]:
        """Check if generated data matches documented assumptions"""
        stats = data.statistics
        validation = {}
        
        for strategy in ["sms", "email", "delayed_retry", "support"]:
            if strategy in stats.get("by_strategy", {}):
                actual = stats["by_strategy"][strategy]["recovery_rate"]
                expected = ASSUMPTIONS["recovery_rates_by_strategy"][strategy]["rate"]
                validation[f"{strategy}_recovery_realistic"] = (expected * 0.7 < actual < expected * 1.3)
        
        return validation


def generate_recoveryflow_dataset(output_file: str = "recoveryflow_synthetic_data.json"):
    """Generate and export realistic RecoveryFlow dataset"""
    generator = RealisticDataGenerator(seed=42, n_merchants=50, n_customers_per_merchant=200)
    
    print("Generating realistic payment recovery dataset...")
    data = generator.generate()
    
    print(f"Generated {data.metadata['total_failures']} payment failures")
    print(f"Overall recovery rate: {data.statistics['overall_recovery_rate']:.1%}")
    print(f"Revenue at risk: ₹{data.statistics['total_revenue_at_risk']:,.0f}")
    print(f"Revenue recovered: ₹{data.statistics['total_revenue_recovered']:,.0f}")
    
    validation = generator.validate_assumptions(data)
    print(f"\nAssumptions validation: {sum(validation.values())}/{len(validation)} checks passed")
    
    generator.export_to_json(output_file)
    print(f"Dataset exported to {output_file}")
    
    return data


if __name__ == "__main__":
    generate_recoveryflow_dataset()
