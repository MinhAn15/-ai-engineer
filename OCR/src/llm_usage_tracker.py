"""
Track LLM API usage and costs
"""
import json
from pathlib import Path
from datetime import datetime


class UsageTracker:
    def __init__(self, log_file='outputs/llm_usage.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        self.usage_data = self._load()
    
    def _load(self):
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {'requests': [], 'total_tokens': 0}
        return {'requests': [], 'total_tokens': 0}
    
    def log_request(self, model, input_tokens, output_tokens, schema):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'input_tokens': int(input_tokens),
            'output_tokens': int(output_tokens),
            'total_tokens': int(input_tokens + output_tokens),
            'schema': schema[:50]  # First 50 chars
        }
        
        self.usage_data['requests'].append(entry)
        self.usage_data['total_tokens'] += entry['total_tokens']
        
        self._save()
        
        # Print stats
        print(f"   💰 LLM Usage: {int(input_tokens):,} in + {int(output_tokens):,} out = {entry['total_tokens']:,} tokens")
        
        # Check if approaching limits
        if self.usage_data['total_tokens'] > 900_000:
            print(f"   ⚠ Warning: Approaching daily free tier limit ({self.usage_data['total_tokens']:,}/1M tokens)")
    
    def _save(self):
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.usage_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"   Warning: Could not save usage data: {e}")
    
    def get_stats(self):
        total_requests = len(self.usage_data['requests'])
        total_tokens = self.usage_data['total_tokens']
        
        # Gemini pricing (approximate)
        # Free tier: 15 RPM, 1M tokens/day
        # Paid: $0.00025/1k input tokens, $0.0005/1k output tokens
        
        # Calculate approximate cost if paid
        total_input = sum(r['input_tokens'] for r in self.usage_data['requests'])
        total_output = sum(r['output_tokens'] for r in self.usage_data['requests'])
        
        cost_input = (total_input / 1000) * 0.00025
        cost_output = (total_output / 1000) * 0.0005
        total_cost = cost_input + cost_output
        
        return {
            'total_requests': total_requests,
            'total_tokens': total_tokens,
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'estimated_cost_usd': round(total_cost, 4),
            'free_tier_ok': total_tokens < 1_000_000,  # Daily limit
            'free_tier_usage_pct': round((total_tokens / 1_000_000) * 100, 2)
        }
    
    def print_summary(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("📊 LLM USAGE SUMMARY")
        print("="*60)
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print(f"  Input: {stats['total_input_tokens']:,}")
        print(f"  Output: {stats['total_output_tokens']:,}")
        print(f"Free tier usage: {stats['free_tier_usage_pct']}% of daily limit")
        print(f"Estimated cost (if paid): ${stats['estimated_cost_usd']:.4f}")
        print("="*60)


if __name__ == '__main__':
    tracker = UsageTracker()
    tracker.print_summary()
