"""
Visualize Trading Council debates and decisions
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CouncilVisualizer:
    """Visualize and format council debates for analysis"""
    
    def __init__(self):
        self.output_dir = Path("reports/council_debates")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_agent_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format individual agent analysis"""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Agent: {analysis['agent_type']}")
        output.append(f"Recommendation: {analysis['recommendation']} (Confidence: {analysis['confidence']}%)")
        output.append(f"{'='*60}")
        
        if analysis.get('reasoning'):
            output.append("\nReasoning:")
            for i, reason in enumerate(analysis['reasoning'], 1):
                output.append(f"  {i}. {reason}")
        
        if analysis.get('concerns'):
            output.append("\nConcerns:")
            for concern in analysis['concerns']:
                output.append(f"  - {concern}")
        
        if analysis.get('entry_price'):
            output.append(f"\nPrice Levels:")
            output.append(f"  Entry: {analysis['entry_price']}")
            output.append(f"  Stop Loss: {analysis.get('stop_loss', 'N/A')}")
            output.append(f"  Take Profit: {analysis.get('take_profit', 'N/A')}")
        
        return "\n".join(output)
    
    def format_debate_round(self, responses: List[Dict[str, Any]], round_num: int) -> str:
        """Format a debate round"""
        output = []
        output.append(f"\n{'#'*60}")
        output.append(f"DEBATE ROUND {round_num}")
        output.append(f"{'#'*60}")
        
        for response in responses:
            if response['round'] == round_num:
                output.append(f"\n[{response['agent_type']}]:")
                output.append(f"{response['statement']}")
                
                if response.get('updated_confidence') is not None:
                    output.append(f"(Updated confidence: {response['updated_confidence']}%)")
                
                if not response.get('maintains_position', True):
                    output.append("*** POSITION CHANGED ***")
        
        return "\n".join(output)
    
    def format_final_decision(self, decision: Dict[str, Any]) -> str:
        """Format the final council decision"""
        output = []
        output.append(f"\n{'*'*60}")
        output.append("FINAL COUNCIL DECISION")
        output.append(f"{'*'*60}")
        
        signal = decision.get('signal', {})
        output.append(f"\nSymbol: {signal.get('symbol', 'N/A')}")
        output.append(f"Decision: {signal.get('signal', 'N/A')}")
        output.append(f"Overall Confidence: {decision.get('final_confidence', 0):.1f}%")
        output.append(f"  - LLM Confidence: {decision.get('llm_confidence', 0):.1f}%")
        output.append(f"  - ML Confidence: {decision.get('ml_confidence', 0):.1f}%")
        output.append(f"Consensus Level: {decision.get('consensus_level', 0):.1f}%")
        output.append(f"Dissenting Views: {decision.get('dissent_count', 0)}")
        
        if signal.get('rationale'):
            output.append(f"\nRationale: {signal['rationale']}")
        
        if signal.get('entry_price'):
            output.append(f"\nExecution Details:")
            output.append(f"  Entry: {signal['entry_price']}")
            output.append(f"  Stop Loss: {signal.get('stop_loss', 'N/A')}")
            output.append(f"  Take Profit: {signal.get('take_profit', 'N/A')}")
            output.append(f"  Risk Class: {signal.get('risk_class', 'N/A')}")
        
        return "\n".join(output)
    
    def generate_debate_report(
        self,
        symbol: str,
        agent_analyses: List[Dict[str, Any]],
        debate_log: List[Dict[str, Any]],
        final_decision: Dict[str, Any]
    ) -> str:
        """Generate complete debate report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"council_debate_{symbol}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        report = []
        report.append(f"TRADING COUNCIL DEBATE REPORT")
        report.append(f"Symbol: {symbol}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"{'='*80}\n")
        
        # Phase 1: Individual Analyses
        report.append("PHASE 1: INDIVIDUAL AGENT ANALYSES")
        report.append("="*80)
        
        for analysis in agent_analyses:
            report.append(self.format_agent_analysis(analysis))
        
        # Phase 2: Debate
        report.append("\n\nPHASE 2: COUNCIL DEBATE")
        report.append("="*80)
        
        for round_num in range(1, 4):
            round_responses = [r for r in debate_log if r['round'] == round_num]
            if round_responses:
                report.append(self.format_debate_round(round_responses, round_num))
        
        # Phase 3: Final Decision
        report.append("\n\nPHASE 3: FINAL DECISION")
        report.append("="*80)
        report.append(self.format_final_decision(final_decision))
        
        # Summary statistics
        report.append(f"\n\n{'='*80}")
        report.append("SUMMARY STATISTICS")
        report.append("="*80)
        
        # Vote breakdown
        votes = {}
        for analysis in agent_analyses:
            rec = analysis['recommendation']
            votes[rec] = votes.get(rec, 0) + 1
        
        report.append("\nVote Breakdown:")
        for rec, count in votes.items():
            percentage = (count / len(agent_analyses)) * 100
            report.append(f"  {rec}: {count} votes ({percentage:.0f}%)")
        
        # Confidence distribution
        confidences = [a['confidence'] for a in agent_analyses]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        report.append(f"\nConfidence Distribution:")
        report.append(f"  Average: {avg_confidence:.1f}%")
        report.append(f"  Highest: {max(confidences)}% ({[a['agent_type'] for a in agent_analyses if a['confidence'] == max(confidences)][0]})")
        report.append(f"  Lowest: {min(confidences)}% ({[a['agent_type'] for a in agent_analyses if a['confidence'] == min(confidences)][0]})")
        
        # Save report
        full_report = "\n".join(report)
        filepath.write_text(full_report)
        
        logger.info(f"Debate report saved to: {filepath}")
        return full_report
    
    def generate_summary_dashboard(self, recent_decisions: List[Dict[str, Any]]) -> str:
        """Generate a summary dashboard of recent council decisions"""
        
        dashboard = []
        dashboard.append("TRADING COUNCIL DASHBOARD")
        dashboard.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dashboard.append("="*80)
        
        if not recent_decisions:
            dashboard.append("\nNo recent council decisions to display.")
            return "\n".join(dashboard)
        
        dashboard.append(f"\nRecent Decisions ({len(recent_decisions)} total):")
        dashboard.append("-"*80)
        
        for decision in recent_decisions:
            dashboard.append(f"\nTime: {decision['timestamp']}")
            dashboard.append(f"Symbol: {decision['symbol']}")
            dashboard.append(f"Decision: {decision['decision']} (Confidence: {decision['confidence']:.0f}%)")
            dashboard.append(f"Consensus: {decision['consensus']:.0f}%")
            
            # Agent votes
            votes = decision.get('agent_votes', {})
            if votes:
                dashboard.append("Agent Votes:")
                for agent, vote in votes.items():
                    dashboard.append(f"  {agent}: {vote}")
            
            dashboard.append("-"*40)
        
        # Overall statistics
        dashboard.append("\n" + "="*80)
        dashboard.append("OVERALL STATISTICS")
        dashboard.append("="*80)
        
        # Decision distribution
        decisions = {}
        for d in recent_decisions:
            dec = d['decision']
            decisions[dec] = decisions.get(dec, 0) + 1
        
        dashboard.append("\nDecision Distribution:")
        for dec, count in decisions.items():
            percentage = (count / len(recent_decisions)) * 100
            dashboard.append(f"  {dec}: {count} ({percentage:.0f}%)")
        
        # Average metrics
        avg_confidence = sum(d['confidence'] for d in recent_decisions) / len(recent_decisions)
        avg_consensus = sum(d['consensus'] for d in recent_decisions) / len(recent_decisions)
        
        dashboard.append(f"\nAverage Metrics:")
        dashboard.append(f"  Confidence: {avg_confidence:.1f}%")
        dashboard.append(f"  Consensus: {avg_consensus:.1f}%")
        
        return "\n".join(dashboard)


async def visualize_latest_council_decision():
    """Visualize the most recent council decision"""
    
    try:
        from config.settings import get_settings
        from trading_loop import DependencyContainer
        
        settings = get_settings()
        container = DependencyContainer(settings)
        signal_service = container.signal_service()
        
        # Get recent council history
        history = await signal_service.get_council_history(limit=1)
        
        if not history:
            logger.info("No council decisions found.")
            return
        
        latest = history[0]
        logger.info(f"Visualizing council decision from {latest['timestamp']}")
        
        # Create visualizer
        visualizer = CouncilVisualizer()
        
        # Generate dashboard
        dashboard = visualizer.generate_summary_dashboard([latest])
        print("\n" + dashboard)
        
        # Note: Full debate visualization would require storing the complete
        # agent analyses and debate log in the council history
        
    except Exception as e:
        logger.error(f"Error visualizing council decision: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(visualize_latest_council_decision())