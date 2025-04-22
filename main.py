import argparse
import random
import time
from agents.agent2 import MinimaxPlayer
from agents.board_reading_agent import BoardReadingAgent
from agents.strategic_sacrifice import StrategicSacrificeAgent
from agents.edge_control_agent import EdgeControlAgent
from agents.BridgeAgent import BridgeAgent
from agents.TemplatePattern import TemplatePatternAgent
from agents.ThreatCreation import ThreatCreationAgent
from agents.virtual_connect import VirtualConnectionAgent
from agents.DefensivePosition import DefensivePositioningAgent
from agents.opening_stratergy import OpeningStrategyAgent
from agents.CenterDominance import CenterDominanceAgent
from core.controller import Controller
from gui.gui import GUI
from agents.agent1 import CustomPlayer
from agents.agent import Agent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_single_game(size, player1_type, player2_type):
    player_classes = {
        "Agent": CustomPlayer, 
        "minimax": MinimaxPlayer,
        "board_reader": BoardReadingAgent,  # Add the new agent to the dictionary
        "StrategicSacrifice": StrategicSacrificeAgent,
        "EdgeControl": EdgeControlAgent,
        "BridgeAgent": BridgeAgent,
        "TemplatePattern": TemplatePatternAgent,
        "ThreatCreation": ThreatCreationAgent,
        "VirtualConnection": VirtualConnectionAgent,
        "DefensivePosition": DefensivePositioningAgent,
        "OpeningStrategy": OpeningStrategyAgent,
        "CenterDominance": CenterDominanceAgent

    }
    player1 = player_classes[player1_type](size, 1, 2)
    player2 = player_classes[player2_type](size, 2, 1)
    
    controller = Controller(size, player1, player2)
    
    while controller._winner == 0:
        controller.update()
    
    return {
        "winner": controller._winner,
        "winner_name": player1.name if controller._winner == 1 else player2.name,
        "player1_type": player1_type,
        "player2_type": player2_type
    }
def run_evaluation(num_games=25, size=5):
    print(f"\n{'=' * 60}")
    print(f"EVALUATION MODE: Running {num_games} games with board size {size}")
    print(f"{'=' * 60}\n")
    
    results = []
    player_types = [
        "Agent", "minimax", "board_reader", "StrategicSacrifice", 
        "EdgeControl", "BridgeAgent", "TemplatePattern", "ThreatCreation", 
        "VirtualConnection", "DefensivePosition", "OpeningStrategy", "CenterDominance"
    ]
    
    # Initialize stats dictionary with counters for all agent types
    stats = {"total_games": num_games}
    
    # Initialize win counters for each agent type
    for agent_type in player_types:
        stats[f"{agent_type}_wins"] = 0
        stats[f"{agent_type}_as_p1_wins"] = 0
        stats[f"{agent_type}_as_p2_wins"] = 0
    
    start_time = time.time()
    
    for i in range(num_games):
        # Randomly select two different player types
        player1_type = random.choice(player_types)
        player2_type = random.choice([p for p in player_types if p != player1_type])
        
        print(f"Game {i+1}/{num_games}: Player 1 = {player1_type.capitalize()}, Player 2 = {player2_type.capitalize()}")
        
        result = run_single_game(size, player1_type, player2_type)
        results.append(result)
        
        # Update statistics
        winner_type = result["player1_type"] if result["winner"] == 1 else result["player2_type"]
        stats[f"{winner_type}_wins"] += 1
        if result["winner"] == 1:
            stats[f"{winner_type}_as_p1_wins"] += 1
        else:
            stats[f"{winner_type}_as_p2_wins"] += 1
                
        print(f"  → Winner: Player {result['winner']} ({result['winner_name']})\n")
    
    elapsed_time = time.time() - start_time
    
    # Display comprehensive statistics for all agent types
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 60}")
    
    # Calculate win percentages for all agent types
    for agent_type in player_types:
        # Calculate games played as each position
        games_as_p1 = sum(1 for r in results if r["player1_type"] == agent_type)
        games_as_p2 = sum(1 for r in results if r["player2_type"] == agent_type)
        total_games_played = games_as_p1 + games_as_p2
        
        # Calculate win percentages
        total_wins = stats[f"{agent_type}_wins"]
        win_pct = (total_wins / total_games_played) * 100 if total_games_played > 0 else 0
        
        # Calculate win percentages by position
        p1_win_pct = (stats[f"{agent_type}_as_p1_wins"] / games_as_p1) * 100 if games_as_p1 > 0 else 0
        p2_win_pct = (stats[f"{agent_type}_as_p2_wins"] / games_as_p2) * 100 if games_as_p2 > 0 else 0
        
        print(f"\n{agent_type.capitalize()} Statistics:")
        print(f"  Games played: {total_games_played}")
        print(f"  Total wins: {total_wins}/{total_games_played} ({win_pct:.1f}%)")
        print(f"  As Player 1: {stats[f'{agent_type}_as_p1_wins']}/{games_as_p1} wins ({p1_win_pct:.1f}%)")
        print(f"  As Player 2: {stats[f'{agent_type}_as_p2_wins']}/{games_as_p2} wins ({p2_win_pct:.1f}%)")
    
    print(f"\n{'=' * 60}")
    print(f"Total games played: {stats['total_games']}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per game: {elapsed_time / stats['total_games']:.2f} seconds")
    print(f"{'=' * 60}\n")
    
    return stats
def main():
    parser = argparse.ArgumentParser(description="AI-Rena Game")
    parser.add_argument("--gui", type=str2bool, default=True, help="Enable GUI (true/false)")
    # Update choices to include the new agent type
    parser.add_argument("--players", nargs=2, choices=["Agent", "minimax", "board_reader", "StrategicSacrifice", "EdgeControl", "BridgeAgent","TemplatePattern", "ThreatCreation", "VirtualConnection", "DefensivePosition", "OpeningStrategy", "CenterDominance"],
                        required=False, help="Player types")
    parser.add_argument("--size", type=int, default=9, help="Grid size")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    args = parser.parse_args()
    
    # If eval mode is enabled, run the evaluation and exit
    if args.eval:
        run_evaluation(num_games=250, size=5)
        return
    
    # Regular game mode
    if not args.players:
        parser.error("the --players argument is required for regular game mode")
    
    player_classes = {
                "Agent": CustomPlayer, 
        "minimax": MinimaxPlayer,
        "board_reader": BoardReadingAgent,  # Add the new agent to the dictionary
        "StrategicSacrifice": StrategicSacrificeAgent,
        "EdgeControl": EdgeControlAgent,
        "BridgeAgent": BridgeAgent,
        "TemplatePattern": TemplatePatternAgent,
        "ThreatCreation": ThreatCreationAgent,
        "VirtualConnection": VirtualConnectionAgent,
        "DefensivePosition": DefensivePositioningAgent,
        "OpeningStrategy": OpeningStrategyAgent,
        "CenterDominance": CenterDominanceAgent
    }
    player1 = player_classes[args.players[0]](args.size, 1, 2)
    player2 = player_classes[args.players[1]](args.size, 2, 1)

    controller = Controller(args.size, player1, player2)
    if args.gui:
        gui = GUI(controller)
        gui.start()
        gui.run()
    else:
        while controller._winner == 0:
            controller.update()
        print(f"Player {controller._winner} wins!")

if __name__ == "__main__":
    main()