# test_model_performance.py

import time
import asyncio
import statistics
from typing import List, Dict, Any
import pytest
from src.model import ModelInterface, get_model_interface
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class ModelPerformanceTest:
    """Test suite for model performance analysis."""
    
    def __init__(self):
        self.model = get_model_interface()
        self.test_queries = [
            "What is a black hole?",
            "Explain quantum entanglement",
            "How does neural networking work?",
            "Describe the theory of relativity",
            "What is dark matter?"
        ]
        self.results = {}
        
    async def test_latency(self, num_iterations: int = 5) -> Dict[str, float]:
        """Test model latency with multiple queries."""
        latencies = []
        token_counts = []
        
        print("\nRunning Latency Test...")
        for i in range(num_iterations):
            for query in self.test_queries:
                start_time = time.time()
                
                messages = [{"role": "user", "content": query}]
                response = self.model.interact_with_model(messages)
                
                if response:
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    # Calculate tokens
                    if 'usage' in response:
                        token_counts.append(
                            response['usage'].get('total_tokens', 0)
                        )
                
                print(f"Query {i+1}: Latency = {latency:.2f}s")
                
        results = {
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies),
            "avg_tokens": statistics.mean(token_counts) if token_counts else 0
        }
        
        self.results["latency"] = results
        return results

    async def test_throughput(self, batch_size: int = 5, num_batches: int = 3) -> Dict[str, float]:
        """Test model throughput with concurrent requests."""
        print("\nRunning Throughput Test...")
        
        async def process_batch(queries: List[str]) -> List[float]:
            start_time = time.time()
            tasks = []
            
            for query in queries:
                messages = [{"role": "user", "content": query}]
                tasks.append(
                    self.model.interact_with_model(messages)
                )
                
            responses = await asyncio.gather(*tasks)
            batch_time = time.time() - start_time
            
            return batch_time, len([r for r in responses if r])

        total_queries = 0
        total_time = 0
        successful_queries = 0
        
        for i in range(num_batches):
            batch_queries = self.test_queries * (batch_size // len(self.test_queries) + 1)
            batch_queries = batch_queries[:batch_size]
            
            batch_time, success_count = await process_batch(batch_queries)
            
            total_time += batch_time
            total_queries += batch_size
            successful_queries += success_count
            
            print(f"Batch {i+1}: Processed {success_count}/{batch_size} queries in {batch_time:.2f}s")
            
        results = {
            "queries_per_second": total_queries / total_time,
            "success_rate": successful_queries / total_queries,
            "avg_batch_time": total_time / num_batches
        }
        
        self.results["throughput"] = results
        return results

    async def test_token_optimization(self) -> Dict[str, Any]:
        """Test token usage and optimization opportunities."""
        print("\nRunning Token Optimization Test...")
        
        test_cases = [
            {
                "name": "Short query",
                "content": "What is a black hole?"
            },
            {
                "name": "Medium query",
                "content": "Explain the relationship between quantum mechanics and black holes."
            },
            {
                "name": "Long query with context",
                "content": "Considering the recent developments in astrophysics and the first "
                          "images of black holes, how has our understanding of black hole physics "
                          "evolved over the past decade?"
            }
        ]
        
        results = []
        for test in test_cases:
            messages = [{"role": "user", "content": test["content"]}]
            start_time = time.time()
            response = self.model.interact_with_model(messages)
            
            if response and 'usage' in response:
                results.append({
                    "test_name": test["name"],
                    "input_length": len(test["content"]),
                    "input_tokens": response['usage'].get('prompt_tokens', 0),
                    "output_tokens": response['usage'].get('completion_tokens', 0),
                    "total_tokens": response['usage'].get('total_tokens', 0),
                    "latency": time.time() - start_time
                })
                
                print(f"{test['name']}:")
                print(f"  Input Length: {len(test['content'])} chars")
                print(f"  Total Tokens: {results[-1]['total_tokens']}")
                print(f"  Latency: {results[-1]['latency']:.2f}s")
        
        self.results["token_optimization"] = results
        return results

    def generate_report(self):
        """Generate comprehensive performance report."""
        if not self.results:
            return "No test results available."
            
        report = ["Model Performance Report", "=" * 50, ""]
        
        if "latency" in self.results:
            report.extend([
                "Latency Analysis:",
                "-" * 20,
                f"Average Latency: {self.results['latency']['avg_latency']:.2f}s",
                f"Min Latency: {self.results['latency']['min_latency']:.2f}s",
                f"Max Latency: {self.results['latency']['max_latency']:.2f}s",
                f"Standard Deviation: {self.results['latency']['std_dev']:.2f}s",
                f"Average Tokens: {self.results['latency']['avg_tokens']:.1f}",
                ""
            ])
            
        if "throughput" in self.results:
            report.extend([
                "Throughput Analysis:",
                "-" * 20,
                f"Queries per Second: {self.results['throughput']['queries_per_second']:.2f}",
                f"Success Rate: {self.results['throughput']['success_rate']*100:.1f}%",
                f"Average Batch Time: {self.results['throughput']['avg_batch_time']:.2f}s",
                ""
            ])
            
        if "token_optimization" in self.results:
            report.extend([
                "Token Usage Analysis:",
                "-" * 20
            ])
            
            for test in self.results["token_optimization"]:
                report.extend([
                    f"\nTest: {test['test_name']}",
                    f"Input Length: {test['input_length']} chars",
                    f"Input Tokens: {test['input_tokens']}",
                    f"Output Tokens: {test['output_tokens']}",
                    f"Total Tokens: {test['total_tokens']}",
                    f"Latency: {test['latency']:.2f}s"
                ])
                
        return "\n".join(report)

async def main():
    """Run performance tests and generate report."""
    tester = ModelPerformanceTest()
    
    # Run all tests
    await tester.test_latency()
    await tester.test_throughput()
    await tester.test_token_optimization()
    
    # Generate and print report
    report = tester.generate_report()
    print("\n" + report)

if __name__ == "__main__":
    asyncio.run(main())