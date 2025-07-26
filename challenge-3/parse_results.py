"""
Evaluation Results Parser

This script parses the evaluation output JSON file and displays the most important 
information in a readable format, including metrics, performance data, and agent responses.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return {}

def extract_query_info(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract query information from a row"""
    query_info = {}
    
    # Extract user query
    if 'inputs.query' in row:
        for message in row['inputs.query']:
            if message.get('role') == 'user':
                content = message.get('content', [])
                if content and isinstance(content, list):
                    query_info['user_query'] = content[0].get('text', 'N/A')
                break
    
    # Extract agent response
    if 'inputs.response' in row:
        agent_responses = []
        for message in row['inputs.response']:
            if message.get('role') == 'assistant':
                content = message.get('content', [])
                for item in content:
                    if item.get('type') == 'text':
                        agent_responses.append(item.get('text', ''))
        query_info['agent_response'] = '\n'.join(agent_responses) if agent_responses else 'N/A'
    
    # Extract tool usage information
    tool_calls = 0
    for message in row.get('inputs.response', []):
        if message.get('role') == 'assistant':
            content = message.get('content', [])
            for item in content:
                if item.get('type') == 'tool_call':
                    tool_calls += 1
    query_info['tool_calls_made'] = tool_calls
    
    return query_info

def extract_performance_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance metrics from a row"""
    metrics = {}
    
    # Operational metrics
    if 'inputs.metrics' in row:
        input_metrics = row['inputs.metrics']
        metrics['server_duration'] = input_metrics.get('server-run-duration-in-seconds', 0)
        metrics['client_duration'] = input_metrics.get('client-run-duration-in-seconds', 0)
        metrics['completion_tokens'] = input_metrics.get('completion-tokens', 0)
        metrics['prompt_tokens'] = input_metrics.get('prompt-tokens', 0)
        metrics['ground_truth'] = input_metrics.get('ground-truth', 'N/A')
    
    # Evaluation scores
    metrics['intent_resolution_score'] = row.get('outputs.intent_resolution.intent_resolution', 'N/A')
    metrics['intent_resolution_result'] = row.get('outputs.intent_resolution.intent_resolution_result', 'N/A')
    metrics['task_adherence_score'] = row.get('outputs.task_adherence.task_adherence', 'N/A')
    metrics['task_adherence_result'] = row.get('outputs.task_adherence.task_adherence_result', 'N/A')
    
    return metrics

def format_duration(seconds: float) -> str:
    """Format duration in a readable way"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

def print_summary_stats(rows: List[Dict[str, Any]]):
    """Print overall summary statistics"""
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    total_queries = len(rows)
    total_server_time = sum(row.get('inputs.metrics', {}).get('server-run-duration-in-seconds', 0) for row in rows)
    total_client_time = sum(row.get('inputs.metrics', {}).get('client-run-duration-in-seconds', 0) for row in rows)
    total_completion_tokens = sum(row.get('inputs.metrics', {}).get('completion-tokens', 0) for row in rows)
    total_prompt_tokens = sum(row.get('inputs.metrics', {}).get('prompt-tokens', 0) for row in rows)
    
    # Calculate averages for evaluation scores
    intent_scores = [row.get('outputs.intent_resolution.intent_resolution') for row in rows if row.get('outputs.intent_resolution.intent_resolution') is not None]
    task_scores = [row.get('outputs.task_adherence.task_adherence') for row in rows if row.get('outputs.task_adherence.task_adherence') is not None]
    
    avg_intent_score = sum(intent_scores) / len(intent_scores) if intent_scores else 0
    avg_task_score = sum(task_scores) / len(task_scores) if task_scores else 0
    
    print(f"Total Queries Evaluated: {total_queries}")
    print(f"Average Server Response Time: {format_duration(total_server_time / total_queries if total_queries > 0 else 0)}")
    print(f"Average Client Response Time: {format_duration(total_client_time / total_queries if total_queries > 0 else 0)}")
    print(f"Total Tokens Used: {total_completion_tokens + total_prompt_tokens:,}")
    print(f"  - Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  - Completion Tokens: {total_completion_tokens:,}")
    print(f"Average Intent Resolution Score: {avg_intent_score:.1f}/5")
    print(f"Average Task Adherence Score: {avg_task_score:.1f}/5")
    print()

def print_detailed_results(rows: List[Dict[str, Any]]):
    """Print detailed results for each query"""
    print("📋 DETAILED QUERY RESULTS")
    print("=" * 60)
    
    for i, row in enumerate(rows, 1):
        query_info = extract_query_info(row)
        metrics = extract_performance_metrics(row)
        
        print(f"\n🔍 Query {i}:")
        print("-" * 40)
        
        # Query and response
        print(f"Question: {query_info.get('user_query', 'N/A')[:100]}...")
        print(f"Ground Truth: {metrics.get('ground_truth', 'N/A')[:100]}...")
        
        # Performance metrics
        print(f"Response Time: {format_duration(metrics.get('server_duration', 0))}")
        print(f"Tokens Used: {metrics.get('prompt_tokens', 0) + metrics.get('completion_tokens', 0):,}")
        print(f"Tool Calls Made: {query_info.get('tool_calls_made', 0)}")
        
        # Evaluation scores
        intent_score = metrics.get('intent_resolution_score', 'N/A')
        task_score = metrics.get('task_adherence_score', 'N/A')
        
        print(f"Intent Resolution: {intent_score}/5 ({metrics.get('intent_resolution_result', 'N/A')})")
        print(f"Task Adherence: {task_score}/5 ({metrics.get('task_adherence_result', 'N/A')})")
        
        # Agent response (truncated)
        agent_response = query_info.get('agent_response', 'N/A')
        if len(agent_response) > 200:
            agent_response = agent_response[:200] + "..."
        print(f"Agent Response: {agent_response}")
        
        print()

def print_performance_analysis(rows: List[Dict[str, Any]]):
    """Print performance analysis"""
    print("⚡ PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Response time analysis
    server_times = [row.get('inputs.metrics', {}).get('server-run-duration-in-seconds', 0) for row in rows]
    client_times = [row.get('inputs.metrics', {}).get('client-run-duration-in-seconds', 0) for row in rows]
    
    print(f"Response Time Statistics:")
    print(f"  Fastest Response: {format_duration(min(server_times) if server_times else 0)}")
    print(f"  Slowest Response: {format_duration(max(server_times) if server_times else 0)}")
    print(f"  Average Response: {format_duration(sum(server_times) / len(server_times) if server_times else 0)}")
    
    # Token usage analysis
    completion_tokens = [row.get('inputs.metrics', {}).get('completion-tokens', 0) for row in rows]
    prompt_tokens = [row.get('inputs.metrics', {}).get('prompt-tokens', 0) for row in rows]
    
    print(f"\nToken Usage Statistics:")
    print(f"  Average Prompt Tokens: {sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0:.0f}")
    print(f"  Average Completion Tokens: {sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0:.0f}")
    
    # Quality scores
    intent_scores = [row.get('outputs.intent_resolution.intent_resolution') for row in rows if row.get('outputs.intent_resolution.intent_resolution') is not None]
    task_scores = [row.get('outputs.task_adherence.task_adherence') for row in rows if row.get('outputs.task_adherence.task_adherence') is not None]
    
    print(f"\nQuality Score Distribution:")
    if intent_scores:
        print(f"  Intent Resolution - Min: {min(intent_scores)}, Max: {max(intent_scores)}, Avg: {sum(intent_scores)/len(intent_scores):.1f}")
    if task_scores:
        print(f"  Task Adherence - Min: {min(task_scores)}, Max: {max(task_scores)}, Avg: {sum(task_scores)/len(task_scores):.1f}")
    
    print()

def main():
    """Main function to parse and display evaluation results"""
    # Get the file path
    current_dir = Path(__file__).parent
    file_path = current_dir / "eval-output-simple.json"
    
    # Allow command line argument for different file
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    
    print(f"📂 Loading evaluation results from: {file_path}")
    print()
    
    # Load the data
    data = load_evaluation_results(str(file_path))
    if not data:
        return
    
    # Extract rows
    rows = data.get('rows', [])
    if not rows:
        print("❌ No evaluation data found in the file")
        return
    
    print(f"✅ Found {len(rows)} evaluation results")
    print()
    
    # Display results in different sections
    print_summary_stats(rows)
    print_performance_analysis(rows)
    print_detailed_results(rows)
    
    print("🎯 KEY INSIGHTS:")
    print("=" * 60)
    
    # Calculate pass rates
    intent_passes = sum(1 for row in rows if row.get('outputs.intent_resolution.intent_resolution_result') == 'pass')
    task_passes = sum(1 for row in rows if row.get('outputs.task_adherence.task_adherence_result') == 'pass')
    
    print(f"✅ Intent Resolution Pass Rate: {intent_passes}/{len(rows)} ({intent_passes/len(rows)*100:.1f}%)")
    print(f"✅ Task Adherence Pass Rate: {task_passes}/{len(rows)} ({task_passes/len(rows)*100:.1f}%)")
    
    # Check if all queries used tools
    total_tool_calls = sum(1 for row in rows if any(
        msg.get('content', [{}])[0].get('type') == 'tool_call' 
        for msg in row.get('inputs.response', []) 
        if msg.get('role') == 'assistant'
    ))
    print(f"🔧 Queries Using Azure AI Search: {total_tool_calls}/{len(rows)} ({total_tool_calls/len(rows)*100:.1f}%)")
    
    print(f"\n📁 Full evaluation details saved in: {file_path}")

if __name__ == "__main__":
    main()
